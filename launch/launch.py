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

"""Entry point for launching RoboBallet experiments."""

from collections.abc import Sequence
import dataclasses
import getpass
import os
from typing import Any

from absl import app
from absl import flags
from absl import logging
from dm_env import specs
import jax
from jax import numpy as jnp
import jraph
import launchpad as lp
from orbax import checkpoint
import reverb
import tensorflow as tf

from agents import actor
from agents import compute_features
from agents import evaluator
from agents import networks
from environments import planning_env
from launch import hparams
from launch import launch_utils
from train import stats
from train import train

FLAGS = flags.FLAGS

_NUM_ACTORS = flags.DEFINE_integer('num_actors', 1,
                                   'Number of actors (for local runs only)')
_NUM_TARGETS = flags.DEFINE_integer('num_targets', 10, 'Number of targets')
_NUM_OBSTACLES = flags.DEFINE_integer(
    'num_obstacles', 30, 'Number of obstacles'
)
_NUM_ROBOTS = flags.DEFINE_integer('num_robots', 4, 'Number of robots (4/8)')
_ARTIFACTS_PREFIX = flags.DEFINE_string(
    'artifacts_prefix',
    '.',
    'Directory prefix for experiment artifacts',
)
_TERMINAL = flags.DEFINE_string('terminal', 'tmux_session',
                                'Terminal to use when running locally')
_EXP_NAME = flags.DEFINE_string('exp_name', 'RoboBallet Experiment',
                                'Experiment name')
_RESTORE_CHECKPOINT = flags.DEFINE_string(
    'restore_checkpoint',
    '',
    'Checkpoint dir to restore',
)
_RESTORE_CHECKPOINT_STEP = flags.DEFINE_integer(
    'restore_checkpoint_step',
    -1,
    'Checkpoint step for restore, latest by default',
)
_PREGEN = flags.DEFINE_bool(
    'pregenerated_start_states',
    True,
    'Whether to use pre-generated start states',
)

_REVERB_CHECKPOINT_TIME_DELTA_MINUTES = 15


# GraphsTuple tensors are defined as a union of several types, some of which
# don't have .shape, and so type checker complains when we try to get their
# shapes. This can be silenced by an assert, and that's what this function does.
def _check_jnp_array(maybe_jnp_array: Any) -> jnp.ndarray:
  assert isinstance(maybe_jnp_array, jnp.ndarray)
  return maybe_jnp_array


def _make_replay_tables(
    config: launch_utils.ReplayConfig,
    example_observation: jraph.GraphsTuple,
    action_spec: specs.BoundedArray,
):
  """Replay tables."""
  observation_signature = {
      'nodes': tf.TensorSpec(
          _check_jnp_array(example_observation.nodes).shape, tf.float32
      ),
      'edges': tf.TensorSpec(
          _check_jnp_array(example_observation.edges).shape, tf.float32
      ),
      'globals': tf.TensorSpec(
          _check_jnp_array(example_observation.globals).shape,
          tf.float32
      ),
  }
  signature = {
      'observation': observation_signature,
      'observation_next': observation_signature,
      'trajectory_stats': {
          field.name: tf.TensorSpec(shape=(), dtype=tf.float32)
          for field in dataclasses.fields(stats.TrajectoryStats)
      },
      # Action taken at this step.
      'action': tf.TensorSpec(action_spec.shape, tf.float32),
      # Reward from taking the action.
      'reward': tf.TensorSpec([], tf.float32),
      # Whether the step is terminal (1) or not (0) AFTER applying the
      # action. We don't have an entry for the final observation.
      'terminal': tf.TensorSpec([], tf.int8),
  }
  limiter = reverb.rate_limiters.SampleToInsertRatio(
      min_size_to_sample=config.replay_min_size,
      samples_per_insert=config.replay_samples_per_insert,
      error_buffer=4*20*128  # 4 megabatches * 20 batches/MB * 128/batch.
  )
  return [
      reverb.Table(
          name=actor.REPLAY_TABLE_NAME,
          sampler=reverb.selectors.Uniform(),
          remover=reverb.selectors.Fifo(),
          max_size=config.replay_max_size,
          rate_limiter=limiter,
          signature=signature,
      )
  ]


def _restore_checkpoint_path(
    checkpoint_dir: str, checkpoint_step: int
) -> None | str:
  """Get restore model checkpoint path if specified."""
  if not checkpoint_dir:
    return None
  snapshot_dirs_list = os.listdir(checkpoint_dir)
  snapshot_list = []
  for snapshot_dir in snapshot_dirs_list:
    try:
      snapshot_list.append(int(snapshot_dir))
    except ValueError:
      pass
  if checkpoint_step == -1:
    step = sorted(snapshot_list, reverse=True)[0]
  else:
    step = _RESTORE_CHECKPOINT_STEP.value
  return f'{_RESTORE_CHECKPOINT.value}/model_checkpoints/{step}'


def make_program(
    configs: hparams.LaunchConfigs,
    checkpoint_dir: str | None,
    starting_checkpoint_dir: str | None,
    tensorboard_log_dir: str | None,
) -> lp.Program:
  """Makes the launchpad program."""
  program = lp.Program('roboballet_experiment')

  actor_config = configs.actor_config
  actor_config = actor_config.replace(num_actors=_NUM_ACTORS.value)
  configs = configs.replace(actor_config=actor_config)

  start_state = None

  # Orbax requires checkpoint_dir and starting_checkpoint_dir to be absolute.
  if checkpoint_dir is not None:
    checkpoint_dir = os.path.abspath(checkpoint_dir)
  if starting_checkpoint_dir is not None:
    starting_checkpoint_dir = os.path.abspath(starting_checkpoint_dir)

  if actor_config.start_states_path is not None:
    start_states = actor.load_start_states(actor_config.start_states_path)
    start_state = start_states[0]

  env = planning_env.PlanningEnv(configs.env_config, start_state=start_state)
  env.reset()
  with jax.default_device(jax.devices('cpu')[0]):
    example_features = compute_features.make_graph_features(
        env.observation(),
        env.observation_spec(),
        configs.model_config.feature_config,
    )

    def priority_tables_fn(
        features=example_features,
        action_spec=env.action_spec(),
        config=configs.replay_config,
    ) -> list[reverb.Table]:
      return _make_replay_tables(config, features, action_spec)

    with program.group('replay'):
      replay_node = lp.ReverbNode(
          priority_tables_fn=priority_tables_fn,
      )
      replay_handle = program.add_node(replay_node)

    policy_network = networks.RoboBalletPolicyNet(config=configs.model_config)
    twin_critic_network = networks.RoboBalletTwinCriticNet(
        config=configs.model_config
    )

    with program.group('learner'):
      learner_handle = program.add_node(
          lp.CourierNode(
              train.LearnerNode,
              train_config=configs.train_config,
              feature_config=configs.model_config.feature_config,
              policy_net=policy_network,
              twin_critic_net=twin_critic_network,
              planning_env_config=configs.env_config,
              learner_id=0,
              reverb_client=replay_handle,
              checkpoint_dir=checkpoint_dir,
              starting_checkpoint_dir=starting_checkpoint_dir,
              tensorboard_log_dir=tensorboard_log_dir
          )
      )

    with program.group('actor'):
      for actor_id in range(configs.actor_config.num_actors):
        program.add_node(
            lp.CourierNode(
                actor.ActorNode,
                actor_config=configs.actor_config,
                env_config=configs.env_config,
                network=policy_network,
                actor_id=actor_id,
                reverb_client=replay_handle,
                learner_client=learner_handle
            )
        )

    configs_for_eval = hparams.update_configs_for_eval(configs)

    with program.group('evaluator'):
      program.add_node(
          lp.CourierNode(
              evaluator.EvaluatorNode,
              evaluator_config=configs_for_eval.eval_config,
              actor_config=configs_for_eval.actor_config,
              env_config=configs_for_eval.env_config,
              network=policy_network,
              evaluator_name='training_dist',
              learner_client=learner_handle,
              tensorboard_log_dir=tensorboard_log_dir
          )
      )

      if configs.env_config.base_config.num_robots in (4, 8):
        static_eval_config = configs_for_eval.eval_config
        static_eval_config = static_eval_config.replace(
            episodes_per_network_step=10,
            evaluate_on_static_targets=True,
        )
        configs_for_static_eval = configs_for_eval.replace(
            eval_config=static_eval_config
        )
        program.add_node(
            lp.CourierNode(
                evaluator.EvaluatorNode,
                evaluator_config=configs_for_static_eval.eval_config,
                actor_config=configs_for_static_eval.actor_config,
                env_config=configs_for_static_eval.env_config,
                network=policy_network,
                evaluator_name='static_dist',
                learner_client=learner_handle,
                tensorboard_log_dir=tensorboard_log_dir
            )
        )

    return program


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  logging.set_stderrthreshold('info')

  experiment_title = _EXP_NAME.value

  artifacts_dir_base = _ARTIFACTS_PREFIX.value

  base_configs = hparams.make_optimised_configs(
      num_targets=_NUM_TARGETS.value,
      num_obstacles=_NUM_OBSTACLES.value,
      num_robots=_NUM_ROBOTS.value,
      pregen=_PREGEN.value,
  )

  checkpointer = checkpoint.StandardCheckpointer()
  if _RESTORE_CHECKPOINT.value:
    base_configs = hparams.restore_configs_from_checkpoint(
        checkpointer,
        _RESTORE_CHECKPOINT.value + '/configs.ckpt',
        base_configs,
    )
  model_checkpoint_path = _restore_checkpoint_path(
      checkpoint_dir=_RESTORE_CHECKPOINT.value,
      checkpoint_step=_RESTORE_CHECKPOINT_STEP.value)
  actor_config = base_configs.actor_config.replace(
      num_actors=_NUM_ACTORS.value
  )
  base_configs = base_configs.replace(actor_config=actor_config)
  program = make_program(
      configs=base_configs,
      checkpoint_dir=artifacts_dir_base + '/model_checkpoints',
      starting_checkpoint_dir=model_checkpoint_path,
      tensorboard_log_dir=artifacts_dir_base + '/tensorboard_log',
  )
  lp.launch(program)


if __name__ == '__main__':
  app.run(main)
