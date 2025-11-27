# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Evaluator loop for RoboBallet."""

import collections
import functools
import pprint
import time

from absl import logging
from flax import struct
from google.protobuf import text_format
import jax
import launchpad as lp
import numpy as np
import tensorboardX

from agents import actor
from agents import networks
from data import data_locator
from environments import data_pb2
from environments import planning_env
from environments import target as target_lib


REPLAY_TABLE_NAME = 'replay'


@functools.cache
def get_predefined_targets(
    num_robots: int, num_targets: int
) -> list[list[target_lib.TargetInfo]]:
  """Returns predefined targets for static evaluation."""
  path = data_locator.get_data_path(
      f'paper_test_tasksets/test_{num_targets}targets.pbtxt'
  )
  # In the 8 robots case, we lifted the cube by 0.6m (z axis), so we have to do
  # the same with targets.
  position_offset = np.array([0.0, 0.0, 0.0], dtype=np.float32)
  assert num_robots in (4, 8)
  if num_robots == 8:
    position_offset = np.array([0.0, 0.0, 0.6], dtype=np.float32)
  with open(path) as f:
    data = data_pb2.PredefinedTasks()
    text_format.Parse(f.read(), data)
  target_sets = []
  for target_set in data.target_sets:
    targets = []
    for target_proto in target_set.targets:
      target_info = target_lib.TargetInfo.from_proto(target_proto)
      target_info = target_info.replace(
          position=target_info.position + position_offset
      )
      targets.append(target_info)
    target_sets.append(targets)
  return target_sets


class EvaluatorConfig(struct.PyTreeNode):
  """Evaluator config."""

  episodes_per_network_step: int = 50
  policy_sigma: float = 0.0
  timestep_s: float = 0.1
  collision_penalty_scaling: float = 15.0
  collision_margin: float = 0.0
  evaluate_on_static_targets: bool = False


class EvaluatorNode:
  """Evaluator LaunchPad node."""

  def __init__(
      self,
      evaluator_config: EvaluatorConfig,
      actor_config: actor.ActorConfig,
      env_config: planning_env.PlanningEnvConfig,
      network: networks.RoboBalletPolicyNet,
      evaluator_name: str,
      learner_client: lp.CourierClient,
      tensorboard_log_dir: str,
  ):
    if tensorboard_log_dir:
      logging.info('TensorBoard log dir: %s', tensorboard_log_dir)
      self._summary_writer = tensorboardX.SummaryWriter(
          log_dir=tensorboard_log_dir,
          filename_suffix=f'.evaluator.{evaluator_name}')
    else:
      self._summary_writer = None

    self._jax_device = jax.devices('cpu')[0]
    self._evaluator_config = evaluator_config
    self._actor_config = actor_config
    self._env_config = env_config

    if self._evaluator_config.evaluate_on_static_targets:
      self._actor_config = self._actor_config.replace(
          start_states_path=None,
      )

    network.config = network.config.update_for_cpu_inference()
    self._network = network
    self._evaluator_name = evaluator_name
    self._learner_client = learner_client
    self._network_variables = None
    self._network_step = 0

  def _update_network_variables(self) -> None:
    """Retrieve new network variables."""
    while not lp.stop_event().is_set():
      f = self._learner_client.futures.get_policy_variables()
      try:
        new_variables, step = f.result(timeout=60.0)
        self._network_variables = jax.device_put(
            new_variables, device=self._jax_device
        )
        self._network_variables = networks.update_params_for_cpu_inference(
            self._network_variables
        )
        self._network_step = step
        logging.info('Fetched policy net params (step %d)', step)
        break
      except TimeoutError:
        logging.warning(
            'Timeout fetching policy net params. '
            'Has the learner been shut down?'
        )
        time.sleep(10)

  def run(self):
    """Entry point for the evaluator LaunchPad node."""

    with jax.default_device(self._jax_device):
      logging.info('Evaluator %s started', self._evaluator_name)

      while not lp.stop_event().is_set():
        self._update_network_variables()

        if self._network_variables is None:
          continue

        if lp.stop_event().is_set():
          break

        logging.info('Started evaluating network step %d', self._network_step)

        stats_sum = collections.defaultdict(float)

        env_configs = []

        num_obstacles = self._env_config.num_obstacles_to_generate

        if self._evaluator_config.evaluate_on_static_targets:
          target_sets = get_predefined_targets(
              self._network.config.num_robots,
              self._env_config.num_targets_to_generate,
          )
          assert (
              len(target_sets)
              == self._evaluator_config.episodes_per_network_step
          )
          for target_set in target_sets:
            base_config = self._env_config.base_config
            env_configs.append(
                self._env_config.replace(
                    base_config=base_config,
                    fixed_targets=target_set,
                    num_obstacles_to_generate=0,
                    num_targets_to_generate=0,
                    obstacles_by_decomposition={'workpiece': num_obstacles},
                )
            )
        else:
          env_configs = [
              self._env_config
          ] * self._evaluator_config.episodes_per_network_step

        for i in range(self._evaluator_config.episodes_per_network_step):
          with jax.default_device(self._jax_device):
            trajectory, trajectory_stats = actor.run_episode(
                actor_config=self._actor_config,
                planning_env_config=env_configs[i],
                feature_config=self._network.config.feature_config,
                network_apply_fn=self._network.apply,
                rng_key=jax.random.PRNGKey(i),
                use_random_actions=False,
                stop_event=lp.stop_event(),
                network_variables_fetch_fn=lambda: self._network_variables,
            )
          del trajectory
          stats_sum['final_score'] += trajectory_stats.final_score
          stats_sum['num_targets_done'] += trajectory_stats.num_targets_done
          stats_sum['episode_length'] += trajectory_stats.episode_length
          stats_sum[
              'total_collision_robot_steps'
          ] += trajectory_stats.total_collision_robot_steps
          stats_sum[
              'total_collision_penalty'
          ] += trajectory_stats.total_collision_penalty
          stats_sum[
              'total_mean_acceleration'
          ] += trajectory_stats.total_mean_acceleration
          logging.info('Episode %d: final_score=%f, targets_done=%d, len=%d',
                       i, trajectory_stats.final_score,
                       trajectory_stats.num_targets_done,
                       trajectory_stats.episode_length)
          if lp.stop_event().is_set():
            break

        if lp.stop_event().is_set():
          logging.info('Evaluator %s: Received stop signal',
                       self._evaluator_name)
        else:
          episodes = self._evaluator_config.episodes_per_network_step

          stats = {k: v / episodes for k, v in stats_sum.items()}

          stats = {'network_step': self._network_step, **stats}
          logging.info('%s', pprint.pformat(stats))

          # Write to TensorBoard.
          if self._summary_writer is not None:
            prefix = f'eval/{self._evaluator_name}/'
            for field, value in stats.items():
              self._summary_writer.add_scalar(prefix + field, value,
                                              global_step=self._network_step)
            self._summary_writer.flush()

          # Normalised metrics.
          norm_targets_done = (
              stats['num_targets_done']
              / self._env_config.num_targets_to_generate
          )
          max_steps = int(
              self._env_config.base_config.episode_timeout_s
              / self._env_config.base_config.timestep_s
          )
          norm_episode_len = stats['episode_length'] / (max_steps + 1)
          norm_num_collisions = (
              stats['total_collision_robot_steps']
              / max_steps
              / self._network.config.num_robots
          )
