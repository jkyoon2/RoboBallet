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

"""Acting loop for RoboBallet."""

from collections.abc import Callable
import copy
import dataclasses
import functools
import pprint
import random
import threading
import time
from typing import Any, Optional

from absl import logging
from dm_env import specs
from flax import struct
import jax
from jax import numpy as jnp
import jraph
import launchpad as lp
import numpy as np
import reverb

from agents import compute_features
from agents import networks
from agents import trajectory as trajectory_lib
from data import data_locator
from environments import data_pb2
from environments import planning_env
from environments import target
from environments import util
from train import stats as stats_lib


REPLAY_TABLE_NAME = 'replay'


POLICY_FETCH_TIMEOUT_S = 60.0


# In an exponential decay schedule, the number of steps per decay
# period is just for making the decay rate number easier to intuitively
# understand, and doesn't actually provide an additional control independent
# from decay rate. Since we are doing Vizier tuning, we just set the steps to a
# constant.
#
# decay_factor = decay_rate ** (step / steps_per_decay)
# any change to steps_per_decay is equivalent to a change in decay_rate:
# decay_factor = (decay_rate ** (1 / steps_per_decay)) ** step
_EXP_STEPS_PER_DECAY = 100000


class ActorConfig(struct.PyTreeNode):
  """Actor config."""

  num_actors: int = 32

  # TD3 hyper-parameters.

  # Exploration noise sigma. Paper default value is 0.1. Higher value prevents
  # policy collapse but slows down training. Let's use a  higher one while we
  # debug.
  policy_sigma: float = 0.7

  # policy_sigma decay per 100,000 steps.
  policy_sigma_decay_rate: float = 0.8

  # How many episodes of random exploration before using the network.
  # The paper uses 25e3 transitions. We have a per-actor setting so it depends
  # on the number of actors, but we will tune this anyways. It's probably not
  # very critical. Assuming 8 actors, 25e3 / 8 actors / 100 steps per episode =
  # 31.25.
  random_exploration_episodes: int = 32

  # If an episode solves less than this many targets, also generate a hindsight
  # episode solving this many targets.
  hindsight_num_targets: int = 0

  # Number of steps per policy param update. 0 to disable (update only between
  # episodes).
  steps_per_param_update: int = 0

  # Repeat the same action this many times (for producing smoother videos
  # without changing the dynamics of the interaction).
  action_repeats: int = 1

  # Path to a StartStates proto to be used for randomly sampling start states.
  start_states_path: Optional[str] = None


def _exp_decay(initial_value: float, decay_rate: float, step: int):
  """Exponential decay from initial_value by decay_rate every 1e5 steps."""
  return initial_value * decay_rate ** (step / _EXP_STEPS_PER_DECAY)


def _linear_interpolate(
    start_value: float,
    end_value: float,
    start_step: int,
    end_step: int,
    step: int,
):
  """Linear interpolation between start_value and end_value."""
  if step <= start_step:
    return start_value
  if step >= end_step:
    return end_value
  v_range = end_value - start_value
  return v_range * (step - start_step) / (end_step - start_step) + start_value


@functools.cache
def load_start_states(path: str) -> list[planning_env.StartState]:
  """Loads start states from a file."""
  resolved_path = data_locator.get_data_path(path)
  start_states_proto = data_pb2.StartStates()
  with open(resolved_path, 'rb') as f:
    start_states_data = f.read()
  start_states_proto.ParseFromString(start_states_data)
  start_states = [
      planning_env.StartState.from_proto(start_state_proto)
      for start_state_proto in start_states_proto.start_states
  ]
  return start_states


@functools.partial(
    jax.jit, static_argnames=['network_apply_fn', 'action_scale']
)
def _sample_action_and_apply_noise(
    observation: jraph.GraphsTuple,
    network_variables: Any,
    network_apply_fn: Callable[[Any, jraph.GraphsTuple], jnp.ndarray],
    action_scale: float,
    noise_sigma: jnp.number,
    rng_key: Any,
) -> jnp.ndarray:
  """Returns an action from the network with the supplied noise."""
  network_action = network_apply_fn(network_variables, observation)  # [-1, 1]

  # The output shouldn't be batched.
  assert network_action.shape[0] == 1

  network_action = network_action[0]
  noise = (
      jax.random.normal(key=rng_key, shape=network_action.shape) * noise_sigma
  )
  noisy_action = network_action + noise
  clipped_noisy_action = jnp.clip(noisy_action, -1.0, 1.0)
  return clipped_noisy_action * action_scale


def run_episode(
    actor_config: ActorConfig,
    feature_config: compute_features.FeatureConfig,
    planning_env_config: planning_env.PlanningEnvConfig,
    network_apply_fn: Callable[[Any, jraph.GraphsTuple], jnp.ndarray],
    rng_key: Any,
    use_random_actions: bool = False,
    stop_event: threading.Event | None = None,
    network_variables_fetch_fn: Callable[..., Any] | None = None,
    network_step: int | None = None,
) -> tuple[trajectory_lib.Trajectory, stats_lib.TrajectoryStats]:
  """Runs one episode to generate one trajectory."""

  sampling_start_time_ns = time.time_ns()

  if actor_config.start_states_path is not None:
    while True:
      start_states = load_start_states(actor_config.start_states_path)
      start_state = random.choice(start_states)
      env = planning_env.PlanningEnv(
          config=planning_env_config, start_state=start_state
      )
      try:
        observation = env.reset().observation
      except planning_env.InvalidStartStateError:
        logging.error('Start state contains collision.')
        continue
      break
  else:
    env = planning_env.PlanningEnv(config=planning_env_config)
    observation = env.reset().observation

  obs_spec = env.observation_spec()
  trajectory = trajectory_lib.Trajectory(
      start_state=env.get_start_state(),
      observations=[],
      actions=[],
      rewards=[],
      target_states=[],
  )

  sampling_duration = (time.time_ns() - sampling_start_time_ns) / 1e9

  stepping_start_time_ns = time.time_ns()

  step = 0

  total_reward = 0.0

  action_scale = planning_env_config.base_config.max_joint_velocity

  last_action = None
  action_repeats_remaining = 0

  # Fetch network variables at the last possible moment, so our network doesn't
  # get stale while we are generating obstacles and targets
  # (during env.reset()).
  network_variables = network_variables_fetch_fn()

  while not env.terminal() and (stop_event is None or not stop_event.is_set()):
    graph_features = compute_features.make_graph_features(
        observation, obs_spec, feature_config)
    rng_key, subkey = jax.random.split(rng_key)
    if last_action is None or action_repeats_remaining == 0:
      # We need a new action.
      if use_random_actions:
        action_shape = env.action_spec().shape
        action = jax.random.uniform(
            key=subkey,
            shape=action_shape,
            minval=-action_scale,
            maxval=action_scale,
        )
      else:
        if network_step is not None:
          sigma = _exp_decay(
              initial_value=actor_config.policy_sigma,
              decay_rate=actor_config.policy_sigma_decay_rate,
              step=network_step,
          )
        else:
          sigma = actor_config.policy_sigma
        action = _sample_action_and_apply_noise(
            observation=graph_features,
            network_variables=network_variables,
            network_apply_fn=network_apply_fn,
            action_scale=action_scale,
            noise_sigma=jnp.float32(sigma),
            rng_key=rng_key,
        )
      last_action = action
      action_repeats_remaining = actor_config.action_repeats - 1
    else:
      action = last_action
      action_repeats_remaining -= 1
    trajectory.observations.append(observation)
    trajectory.actions.append(np.array(action))
    timestep = env.step(np.array(action))
    trajectory.rewards.append(timestep.reward)
    observation = timestep.observation
    total_reward += timestep.reward
    trajectory.target_states.append(env.target_states())

    step += 1
    if (network_variables_fetch_fn is not None and
        actor_config.steps_per_param_update > 0 and
        step % actor_config.steps_per_param_update == 0):
      network_variables = network_variables_fetch_fn()

  # Terminal state only has an observation.
  trajectory.observations.append(env.observation())

  # Doing the subtraction before converting to float avoids issues with float
  # losing precision over time.
  episode_duration = (time.time_ns() - stepping_start_time_ns) / 1e9

  stats = stats_lib.TrajectoryStats(
      final_score=env.current_score(),
      total_reward=total_reward,
      total_collision_penalty=env.total_collision_penalty(),
      total_collision_robot_steps=env.total_collision_robot_steps(),
      total_mean_acceleration=env.total_acceleration(),
      num_targets_done=env.num_targets_done(),
      episode_length=len(trajectory.observations),
      env_sampling_time_s=sampling_duration,
      episode_wall_time_s=episode_duration,
  )

  return trajectory, stats


def generate_hindsight_trajectory(
    actual_trajectory: trajectory_lib.Trajectory,
    actual_stats: stats_lib.TrajectoryStats,
    planning_env_config: planning_env.PlanningEnvConfig,
    num_targets: int,
) -> tuple[trajectory_lib.Trajectory, stats_lib.TrajectoryStats] | None:
  """Returns a hindsight experience with `num_targets` solved.

  Generates and returns a hindsight experience with `num_targets` solved.

  Args:
    actual_trajectory: Actual trajectory to modify targets on.
    actual_stats: Stats for the actual trajectory.
    planning_env_config: Planning env config.
    num_targets: Number of targets to be solved. This includes both real targets
      and targets moved with hindsight.

  Returns:
    New hindsight trajectory with `num_targets` targets solved if the actual
      trajectory has fewer than `num_targets` solved. Otherwise None.
  """

  # 1. We iterate through the episode looking at the observations, to figure out
  #    robot tip poses and get the list of targets solved at each timestep.
  tip_poses = []
  targets_solved = []
  for observation in actual_trajectory.observations:
    base_poses = observation['robots']['base_poses']
    tip_relative_poses = observation['robots']['tip_relative_poses']
    tip_poses.append(np.matmul(base_poses, tip_relative_poses))
    targets_solved_at_this_timestep = []
    targets_done = observation['targets']['done']
    for target_id in range(len(actual_trajectory.start_state.targets)):
      if targets_done[target_id]:
        targets_solved_at_this_timestep.append(target_id)
    targets_solved.append(targets_solved_at_this_timestep)

  # 2. If the real trajectory has `num_targets` solved or more already, return
  #    None as we don't need to make a hindsight episode.
  #    Otherwise we calculate how many additional targets we need to move, and
  #    randomly select those targets among the unsolved ones.
  targets_already_solved = targets_solved[-1]
  num_targets_already_solved = len(targets_already_solved)
  if num_targets_already_solved >= num_targets:
    return None
  all_targets = set(range(len(actual_trajectory.start_state.targets)))
  unsolved_targets = all_targets - set(targets_already_solved)
  target_indices_to_change = random.sample(
      list(unsolved_targets), num_targets - num_targets_already_solved)

  # 3. Create a new start state with moved targets, allocating each newly
  #    "solved" targets to one of the arms, at their tip pose at a random time.
  new_start_state = copy.deepcopy(actual_trajectory.start_state)
  num_arms = tip_poses[0].shape[0]
  for target_idx in target_indices_to_change:
    # Replace target at target_idx by a random arm's position at some random
    # point in time.
    arm = random.randint(0, num_arms - 1)
    solution_step = random.randint(1, len(actual_trajectory.actions) - 1)
    tip_pose = tip_poses[solution_step][arm]
    # Put the target where the tip of the arm is at this random point in time.
    new_target_info = target.TargetInfo(
        position=tip_pose[:3, 3], rotation=tip_pose[:3, :3]
    )
    new_start_state.targets[target_idx] = new_target_info

  # 4. Finally, run the new episode again using this new start state.
  env = planning_env.PlanningEnv(config=planning_env_config,
                                 start_state=new_start_state)
  observation = env.reset().observation

  new_trajectory = trajectory_lib.Trajectory(
      start_state=new_start_state,
      observations=[],
      actions=[],
      rewards=[],
      target_states=[],
  )

  step = 0
  while not env.terminal():
    # We have to recompute features because targets are part of features. But
    # the robot part stays the same, so we can use that to avoid doing collision
    # checking again.
    action = actual_trajectory.actions[step]
    new_trajectory.observations.append(observation)
    new_trajectory.actions.append(action)
    timestep = env.step(
        action,
        post_action_robot_obs=actual_trajectory.observations[
            step + 1]['robots'])
    new_trajectory.rewards.append(timestep.reward)
    new_trajectory.target_states.append(env.target_states())
    observation = timestep.observation
    step += 1

  # Terminal state only has an observation.
  new_trajectory.observations.append(env.observation())

  # We return same stats as the original trajectory (except setting the
  # is_hindsight_replay bit), so that stats computed on the learner makes sense.
  stats = copy.copy(actual_stats)
  stats.is_hindsight_replay = 1

  return new_trajectory, stats


def _make_transition_protos_from_trajectory(
    trajectory: trajectory_lib.Trajectory,
    observation_spec: planning_env.SpecTree,
    action_spec: specs.BoundedArray,
) -> list[bytes]:
  """Returns a list of serialized transition protos from a trajectory."""
  transitions = []
  assert len(trajectory.observations) == len(trajectory.actions) + 1
  for t, _ in enumerate(trajectory.actions):
    scaled_obs = jax.tree_util.tree_map(
        compute_features.scale_array,
        trajectory.observations[t],
        observation_spec,
    )
    scaled_next_obs = jax.tree_util.tree_map(
        compute_features.scale_array,
        trajectory.observations[t + 1],
        observation_spec,
    )
    scaled_action = compute_features.scale_array(
        trajectory.actions[t], action_spec
    )
    reward = trajectory.rewards[t]
    is_terminal = t == len(trajectory.actions) - 1
    transition_proto = data_pb2.Transition(
        scaled_observation=util.pytree_to_proto(scaled_obs),
        scaled_action=util.to_proto(scaled_action),
        scaled_next_observation=util.pytree_to_proto(scaled_next_obs),
        reward=reward,
        is_terminal=is_terminal,
    )
    transitions.append(transition_proto.SerializeToString())
  return transitions


def _write_episode(
    trajectory: trajectory_lib.Trajectory,
    trajectory_stats: stats_lib.TrajectoryStats,
    planning_env_config: planning_env.PlanningEnvConfig,
    observation_spec: planning_env.SpecTree,
    feature_config: compute_features.FeatureConfig,
    reverb_client: reverb.Client,
) -> None:
  """Writes a trajectory to a replay buffer."""

  action_scaling = 1.0 / planning_env_config.base_config.max_joint_velocity

  try:
    with reverb_client.trajectory_writer(
        num_keep_alive_refs=len(trajectory.observations) + 1,
    ) as writer:
      num_observations = len(trajectory.observations)
      num_actions = len(trajectory.actions)
      assert num_observations == num_actions + 1
      last_observation_idx = num_observations - 1

      # Terminal action is the last action (that leads to the terminal state).
      terminal_action_idx = num_actions - 1
      for t in range(num_observations):
        data = {}
        features = compute_features.make_graph_features(
            trajectory.observations[t], observation_spec, feature_config)
        data['observation'] = {
            'nodes': features.nodes,
            'edges': features.edges,
            'globals': features.globals,
        }

        if t < last_observation_idx:
          # If this is not the final observation, add action and reward.
          # Scale action back to [-1, 1] here.
          data['action'] = trajectory.actions[t] * action_scaling
          data['reward'] = np.float32(trajectory.rewards[t])
          data['terminal'] = np.int8(1 if t == terminal_action_idx else 0)

        if t == 0:
          data['trajectory_stats'] = {
              k: np.float32(v)
              for k, v in dataclasses.asdict(trajectory_stats).items()
          }
        writer.append(data)

      # The final observation doesn't get its own item, but is used as
      # observation_next for the last item.
      for t in range(num_actions):
        observation = (
            {
                'nodes': writer.history['observation']['nodes'][t],
                'edges': writer.history['observation']['edges'][t],
                'globals': writer.history['observation']['globals'][t],
            },
        )
        observation_next = (
            {
                'nodes': writer.history['observation']['nodes'][t + 1],
                'edges': writer.history['observation']['edges'][t + 1],
                'globals': writer.history['observation']['globals'][t + 1],
            },
        )
        writer.create_item(
            table=REPLAY_TABLE_NAME,
            priority=1.0,
            trajectory={
                'observation': observation,
                'observation_next': observation_next,
                'trajectory_stats': {
                    field.name: writer.history['trajectory_stats'][field.name][
                        0]
                    for field in dataclasses.fields(stats_lib.TrajectoryStats)
                },
                'action': writer.history['action'][t],
                'reward': writer.history['reward'][t],
                'terminal': writer.history['terminal'][t],
            },
        )
      writer.end_episode()
  except RuntimeError as e:
    logging.warning('Failed to write episode: %s. This is normal during shut '
                    'down', str(e))


class ActorNode:
  """Actor LaunchPad node."""

  def __init__(
      self,
      actor_config: ActorConfig,
      env_config: planning_env.PlanningEnvConfig,
      network: networks.RoboBalletPolicyNet,
      actor_id: int,
      reverb_client: reverb.Client,
      learner_client: lp.CourierClient,
  ):
    network.config = network.config.update_for_cpu_inference()

    # We run the actor on the CPU. Acting is mostly CPU-limited by collision
    # checking, and running the network on the CPU allows us to scale to many
    # more actors.
    self._jax_device = jax.devices('cpu')[0]
    with jax.default_device(self._jax_device):
      self._actor_config = actor_config
      self._env_config = env_config
      self._network = network
      self._actor_id = actor_id
      self._reverb_client = reverb_client
      self._learner_client = learner_client
      self._network_step = 0

  def _fetch_network_variables(self) -> Any:
    """Updates policy network variables and return the new variables."""
    network_variables = None
    while not lp.stop_event().is_set():
      f = self._learner_client.futures.get_policy_variables()
      try:
        new_variables, step = f.result(timeout=POLICY_FETCH_TIMEOUT_S)
        network_variables = jax.device_put(
            new_variables, device=self._jax_device
        )
        network_variables = networks.update_params_for_cpu_inference(
            network_variables
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
    return network_variables

  def run(self):
    """Entry point for the actor LaunchPad node."""

    example_env = planning_env.PlanningEnv(config=self._env_config)
    observation_spec = example_env.observation_spec()

    with jax.default_device(self._jax_device):
      episode_count = 0
      logging.info('Actor %d started', self._actor_id)

      # We only get the write duration once the write is done, so we can only
      # report it with the next write, and have to keep track of it separately.
      last_reverb_write_duration = 0.0

      rng_key = jax.random.PRNGKey(np.random.randint(0, 2**32 - 1))

      while not lp.stop_event().is_set():
        # If we are on network step 0 the learner hasn't started, so don't use
        # the network (doing uniform random actions give us better exploration
        # than using a random weight network). Only in online training mode.
        in_exploration = (
            episode_count < self._actor_config.random_exploration_episodes
            or self._network_step == 0
        )

        rng_key, subkey = jax.random.split(rng_key)
        trajectory, trajectory_stats = run_episode(
            actor_config=self._actor_config,
            feature_config=self._network.config.feature_config,
            planning_env_config=self._env_config,
            network_apply_fn=self._network.apply,
            rng_key=subkey,
            use_random_actions=in_exploration,
            stop_event=lp.stop_event(),
            network_variables_fetch_fn=self._fetch_network_variables,
            network_step=self._network_step,
        )

        trajectory_stats.episode_write_wall_time_s = last_reverb_write_duration
        trajectory_stats.network_step = self._network_step

        write_start_time_ns = time.time_ns()
        _write_episode(
            trajectory=trajectory,
            trajectory_stats=trajectory_stats,
            planning_env_config=self._env_config,
            observation_spec=observation_spec,
            feature_config=self._network.config.feature_config,
            reverb_client=self._reverb_client,
        )

        maybe_hindsight_trajectory_and_stats = generate_hindsight_trajectory(
            actual_trajectory=trajectory,
            actual_stats=trajectory_stats,
            planning_env_config=self._env_config,
            num_targets=self._actor_config.hindsight_num_targets,
        )

        if maybe_hindsight_trajectory_and_stats is not None:
          hindsight_trajectory, hindsight_stats = (
              maybe_hindsight_trajectory_and_stats
          )
          _write_episode(
              trajectory=hindsight_trajectory,
              trajectory_stats=hindsight_stats,
              planning_env_config=self._env_config,
              observation_spec=observation_spec,
              feature_config=self._network.config.feature_config,
              reverb_client=self._reverb_client,
          )

        # We only get the write duration once the write is done, so we can
        # only report it with the next write.
        last_reverb_write_duration = (
            time.time_ns() - write_start_time_ns
        ) / 1e9

        episode_count += 1
        if lp.stop_event().is_set():
          logging.info('Actor %d: Received stop signal', self._actor_id)
        else:
          logging.info(
              'Actor %d: Episode %d done (random=%d)', self._actor_id,
              episode_count, in_exploration
          )
          logging.info('%s', pprint.pformat(trajectory_stats))

    logging.info('Actor %d exiting.', self._actor_id)
