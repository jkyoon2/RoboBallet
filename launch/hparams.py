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

"""Hyper-parameters."""

from typing import Any

from flax import struct

from agents import actor
from agents import evaluator
from agents import networks
from environments import base_env
from environments import planning_env
from launch import launch_utils
from train import train


class LaunchConfigs(struct.PyTreeNode):
  """All configs for a launch."""

  env_config: planning_env.PlanningEnvConfig
  replay_config: launch_utils.ReplayConfig
  model_config: networks.ModelConfig
  actor_config: actor.ActorConfig
  eval_config: evaluator.EvaluatorConfig
  train_config: train.TrainConfig


def make_env_config(
    num_targets: int,
    num_obstacles: int,
    num_robots: int,
    timestep_s: float = 0.1,
) -> planning_env.PlanningEnvConfig:
  """Creates an environment config."""
  world_num_robots = 8 if num_robots > 4 else 4
  world_file_path = f'mujoco_world/{world_num_robots}_pandas_world.xml'

  base_config = base_env.BaseEnvConfig(
      world_file_path=world_file_path,
      episode_timeout_s=1.0 * num_targets + 2.0,
      timestep_s=timestep_s,
      num_robots=num_robots,
  )
  if world_num_robots == 4:
    base_config = base_config.replace(
        robot_use_order=['panda1', 'panda2', 'panda3', 'panda4']
    )
  return planning_env.PlanningEnvConfig(
      base_config=base_config,
      num_obstacles_to_generate=num_obstacles,
      num_targets_to_generate=num_targets,
      targets_have_orientation=True,
  )


def make_optimised_configs(
    num_targets: int,
    num_obstacles: int,
    num_robots: int,
    pregen: bool = True,
) -> LaunchConfigs:
  """Returns optimised hyperparameter configs.

  The optimal values are set as the default values already in the various Config
  dataclasses, but here we make adjustments for numbers of targets and
  obstacles, which affects the optimal value for some settings.

  Args:
    num_targets: Number of targets.
    num_obstacles: Number of obstacles.
    num_robots: Number of robots.
    pregen: Whether to use pre-generated start states.

  Returns:
    Optimized config for the given env setup.
  """

  # -- Environment config --
  env_config = make_env_config(
      num_targets=num_targets, num_obstacles=num_obstacles,
      num_robots=num_robots
  )

  # -- Replay config --
  # Replay size: optimal max replay size seems to scale inverse-proportionally
  #   with number of targets, because of the way graph nets work, when there is
  #   twice as many targets, it's seeing twice as many target nodes, and twice
  #   as many arm-target edges per transition.
  replay_max_size = int(int(2e6) // (num_targets / 10))
  replay_config = launch_utils.ReplayConfig(
      replay_min_size=replay_max_size // 10, replay_max_size=replay_max_size
  )

  # -- Model config --
  model_config = networks.ModelConfig()
  model_config = model_config.replace(num_robots=num_robots)

  # -- Actor config --
  actor_config = actor.ActorConfig()
  # HER Number of targets: number of actual targets * 0.4 seems to work well.
  actor_config = actor_config.replace(
      hindsight_num_targets=int(num_targets * 0.4)
  )

  if pregen:
    start_states_path = (
        f'start_states/{num_robots}r_{num_targets}t_{num_obstacles}o.pb')
    actor_config = actor_config.replace(
        start_states_path=start_states_path,
    )

  # -- Evaluator config --
  eval_config = evaluator.EvaluatorConfig()
  # No setup-specific customisations.

  # -- Train config --
  train_config = train.TrainConfig()
  # No setup-specific customisations.

  return LaunchConfigs(
      env_config=env_config,
      replay_config=replay_config,
      model_config=model_config,
      actor_config=actor_config,
      eval_config=eval_config,
      train_config=train_config,
  )


def update_configs_for_eval(base_config: LaunchConfigs) -> LaunchConfigs:
  """Update configs for evaluation."""
  env_config = base_config.env_config
  replay_config = base_config.replay_config
  model_config = base_config.model_config
  actor_config = base_config.actor_config
  eval_config = base_config.eval_config
  train_config = base_config.train_config

  base_env_config = env_config.base_config
  base_env_config = base_env_config.replace(
      timestep_s=eval_config.timestep_s,
      collision_margin=eval_config.collision_margin,
  )
  env_config = env_config.replace(
      base_config=base_env_config,
      randomize_robot_placement=False,
      collision_penalty_scaling=eval_config.collision_penalty_scaling,
  )
  actor_config = actor_config.replace(
      policy_sigma=eval_config.policy_sigma, random_exploration_episodes=0,
  )
  feature_config = model_config.feature_config
  feature_config = feature_config.replace(max_translation_noise=0.0)
  feature_config = feature_config.replace(max_rotation_noise=0.0)
  model_config = model_config.replace(feature_config=feature_config)

  return LaunchConfigs(
      env_config=env_config,
      replay_config=replay_config,
      model_config=model_config,
      actor_config=actor_config,
      eval_config=eval_config,
      train_config=train_config,
  )


def make_example_config_tree() -> LaunchConfigs:
  """Returns an example config tree that can be used to restore configs."""
  return make_optimised_configs(num_targets=1, num_obstacles=1, num_robots=4)


def human_readable_params(params: dict[str, Any]) -> str:
  params_hr = []
  for field, value in params.items():
    field_hr = field.split('.')[-1]
    params_hr.append(f'{field_hr}={str(value)}')
  return '|'.join(params_hr)


def update_configs(configs: Any, field: str, value: Any) -> Any:
  """Returns configs updated with the given param."""
  components = field.split('.')
  if len(components) == 1:
    # Two possibilities here:
    # 1. configs is a dataclass, and we should use replace().
    # 2. We are actually inside a dict, and should use update().
    if isinstance(configs, dict):
      configs.update({field: value})
      return configs
    else:
      return configs.replace(**{field: value})
  else:
    return configs.replace(**{
        components[0]: update_configs(
            getattr(configs, components[0]), '.'.join(components[1:]), value)})


def updates_from_dict(
    dict_updates: dict[str, Any], prefix: str = ''
) -> list[tuple[str, Any]]:
  """Returns updates from a dict."""
  updates = []
  for k, v in dict_updates.items():
    if isinstance(v, dict):
      updates.extend(updates_from_dict(v, prefix + k + '.'))
    else:
      updates.append((prefix + k, v))
  return updates


def restore_configs_from_checkpoint(
    checkpointer: Any, path: str, base_configs: LaunchConfigs
) -> LaunchConfigs:
  """Restores configs from a checkpoint.

  Restore configs from an orbax checkpoint, but do the conversion to config
  dataclasses ourselves instead of using orbax to do it.
  This way we can support config options that have been added since the config
  was checkpointed, as long as the new options have default values. In effect
  this restore function treats values in the restored config as updates to
  base_configs.

  Args:
    checkpointer: Checkpointer.
    path: Path for the checkpoint.
    base_configs: Configs to update.

  Returns:
    Updated configs.
  """
  updates = updates_from_dict(checkpointer.restore(path))
  configs = base_configs
  for field, value in updates:
    configs = update_configs(configs, field, value)
  return configs
