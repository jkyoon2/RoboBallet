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

"""Base dm_env wrapper for giza world."""

from collections.abc import Mapping
import copy
import dataclasses
import functools
import random
from typing import Any

import dm_env
from dm_env import specs
from flax import struct
import numpy as np

from environments import mujoco_simulator


_DEFAULT_EXCLUDED_OBJECTS = frozenset(('workpiece',))

# This order puts one robot on each rail first, to maximize coverage in
# combination with layout optimisation.
_DEFAULT_ROBOT_USE_ORDER = [
    'panda1',
    'panda2',
    'panda5',
    'panda6',
    'panda3',
    'panda4',
    'panda7',
    'panda8',
]


class BaseEnvConfig(struct.PyTreeNode):
  """Base environment configuration.

  Attributes:
      world_file_path: Path to the serialized giza::World or Mujoco model XML.
      episode_timeout_s: Time before episode termination.
      timestep_s: Simulation time per step.
      max_joint_acceleration: Maximum joint acceleration in rad/s^2.
      max_joint_velocity: Maximum joint velocity in rad/s.
      excluded_objects: Objects to remove at reset.
      collision_margin: Collision margin for all collision checks.
      num_robots: Number of robots to use. None for all.
      robot_use_order: Order of robots to use where num_robots < number of
        robots in the world.
      deterministic_mode: If true, the environment will be fully deterministic.
        Note that this will introduce very slight biases.
  """

  world_file_path: str

  episode_timeout_s: float
  timestep_s: float

  max_joint_acceleration: float = 8.0
  max_joint_velocity: float = 2.175

  excluded_objects: list[str] = dataclasses.field(
      default_factory=lambda: list(_DEFAULT_EXCLUDED_OBJECTS)
  )

  collision_margin: float = 0.0

  num_robots: int | None = None

  robot_use_order: list[str] = dataclasses.field(
      default_factory=lambda: list(_DEFAULT_ROBOT_USE_ORDER)
  )

  giza_world_tcp_frame_suffix: str = '::tool|tcp_frame'

  deterministic_mode: bool = False


_EXCLUDED_ROBOTS = frozenset(('leica',))


# World span (meters) for scaling features.
WORLD_SPAN_M = 10


# Special value indicating that this is a brand new environment that hasn't been
# reset yet.
RESET_ON_NEXT_STEP = -1


# Minimum bounds for pose matrices.
POSE_BOUNDS_MIN = np.array([
    [-1.0] * 3 + [-WORLD_SPAN_M / 2],
    [-1.0] * 3 + [-WORLD_SPAN_M / 2],
    [-1.0] * 3 + [-WORLD_SPAN_M / 2],
    [-1.0] * 4,
])


POSE_BOUNDS_MAX = -POSE_BOUNDS_MIN


ArrayTree = Mapping[str, 'ArrayTree | np.ndarray']


def relative_pose(from_pose: np.ndarray, to_pose: np.ndarray) -> np.ndarray:
  """Calculate a transform that transforms from_pose to to_pose."""
  return np.matmul(np.linalg.inv(from_pose), to_pose)


@functools.cache
def load_world_reference_cached(
    world_file_path: str) -> mujoco_simulator.MujocoWorld:
  return mujoco_simulator.MujocoWorld(
      mujoco_simulator.MujocoWorldConfig(xml_path=world_file_path)
  )


class BaseEnv(dm_env.Environment):
  """Base dm_env wrapper for giza world.

  This class implements the dm_env interface to support using a RoboBallet
  simulator in reinforcement learning.
  """

  def __init__(self, config: BaseEnvConfig):
    self._config = config

    self._pristine_world_for_resets = copy.copy(
        load_world_reference_cached(config.world_file_path)
    )

    all_robots = self._pristine_world_for_resets.GetRobots() - _EXCLUDED_ROBOTS

    num_robots = (
        len(all_robots) if config.num_robots is None else config.num_robots
    )

    filtered_robot_use_order = [
        robot for robot in config.robot_use_order if robot in all_robots
    ]

    robots_to_use = filtered_robot_use_order[:num_robots]
    for robot in all_robots:
      if robot not in robots_to_use:
        self._pristine_world_for_resets.RemoveObject(robot)
    for robot in robots_to_use:
      assert robot in all_robots
    self._robots = robots_to_use

    self._world = copy.copy(self._pristine_world_for_resets)

    assert not self._robots_in_collision()

    self._step = RESET_ON_NEXT_STEP

    self._max_steps = int(config.episode_timeout_s / config.timestep_s)

    self._dof_by_robot = [
        len(self._world.GetJointValues(robot))
        for robot in self._robots
    ]

    # We keep track of current joint velocities to implement acceleration
    # limiting. These are the actual velocities.
    self._current_joint_velocities = np.zeros(shape=(
        len(self._robots), self._dof_by_robot[0]
    ), dtype=np.float32)
    self._joint_lower_limits = np.zeros(shape=(
        len(self._robots), self._dof_by_robot[0]
    ), dtype=np.float32)
    self._joint_upper_limits = np.zeros(shape=(
        len(self._robots), self._dof_by_robot[0]
    ), dtype=np.float32)

    for robot_idx, robot in enumerate(self._robots):
      self._joint_lower_limits[robot_idx, :] = (
          self._world.GetLowerJointPositionLimits(robot))
      self._joint_upper_limits[robot_idx, :] = (
          self._world.GetUpperJointPositionLimits(robot))

    # Robots that were in collision in the last step and had to be stopped.
    self._robots_in_collision_last_step = set()

  def _robots_in_collision(self) -> set[str]:
    return self._world.RobotsInCollision(self._config.collision_margin)

  def _move_robots(self, requested_joint_velocities: np.ndarray):
    """Applies a joint move, respecting acceleration and joint limits.

    Args:
      requested_joint_velocities: Joint velocities requested in rad/s for all
        robots.

    Returns:
      None.
    """
    # Apply acceleration clamping.
    max_joint_velocities_change = (
        self._config.max_joint_acceleration * self._config.timestep_s
    )
    min_new_joint_velocities = (
        self._current_joint_velocities - max_joint_velocities_change)
    max_new_joint_velocities = (
        self._current_joint_velocities + max_joint_velocities_change)
    self._current_joint_velocities = np.clip(
        requested_joint_velocities,
        a_min=min_new_joint_velocities,
        a_max=max_new_joint_velocities,
    )
    q_delta = self._current_joint_velocities * self._config.timestep_s
    old_qpos = np.zeros_like(self._current_joint_velocities)
    for robot_idx, robot in enumerate(self._robots):
      old_qpos[robot_idx, :] = self._world.GetJointValues(robot)
    new_qpos = np.clip(
        old_qpos + q_delta,
        a_min=self._joint_lower_limits,
        a_max=self._joint_upper_limits,
    )
    for robot_idx, robot in enumerate(self._robots):
      self._world.SetJointValues(robot, new_qpos[robot_idx])

    robots_in_collision = self._robots_in_collision()
    robots_already_reset = set()
    iterations = 0
    while robots_in_collision:
      # If we have robots in collision, we randomly select one to rollback.
      # Note that this can theoretically cause another robot to go into
      # collision, if the robot that was moved back is now in position to
      # collide with another robot. We can theoretically do something smarter
      # about that, but it's probably extremely rare and not worth the
      # complexity.
      candidates = robots_in_collision - robots_already_reset
      if self._config.deterministic_mode:
        robot_name_to_reset = sorted(candidates)[0]
      else:
        robot_name_to_reset = random.choice(list(candidates))
      robot_idx = self._robots.index(robot_name_to_reset)
      self._world.SetJointValues(robot_name_to_reset, old_qpos[robot_idx])
      self._current_joint_velocities[robot_idx] = np.zeros_like(
          self._current_joint_velocities[robot_idx])
      robots_already_reset.add(robot_name_to_reset)
      robots_in_collision = self._robots_in_collision()
      iterations += 1
      # If we need to do more rounds than the number of robots, something is
      # wrong.
      assert iterations <= len(self._robots)
    self._robots_in_collision_last_step = robots_already_reset

  def reset_impl(self) -> None:
    """Start a new episode by making a copy of the pristine world."""
    self._world = copy.copy(self._pristine_world_for_resets)
    for obj in self._config.excluded_objects:
      self._world.RemoveObject(obj)
    self._step = 0
    self._current_joint_velocities = np.zeros_like(
        self._current_joint_velocities)

  def reset(self) -> dm_env.TimeStep:
    self.reset_impl()
    return dm_env.restart(self.observation())

  def step_impl(self, action: np.ndarray,
                post_action_robot_obs: ArrayTree | None = None) -> None:
    """Makes an environment step (see dm_env for semantics).

    Args:
      action: action to be taken.
      post_action_robot_obs: optional robot observations after the action.
        If we already have observations after the action (eg. we are replaying
        an episode), supplying the post-action observation here allows us to
        skip the collision checking, making the apply action much faster.

    Returns:
      Timestep.

    """
    self._step += 1

    if post_action_robot_obs is not None:
      post_action_joint_configs = post_action_robot_obs['joint_configurations']
      post_action_joint_velocities = post_action_robot_obs['joint_velocities']
      for i, robot in enumerate(self._robots):
        self._world.SetJointValues(robot, post_action_joint_configs[i])
        self._current_joint_velocities[i] = post_action_joint_velocities[i]
    else:
      self._move_robots(action)

  def step(
      self, action: np.ndarray, allow_collisions: bool = False
  ) -> dm_env.TimeStep:
    if self.terminal():
      return self.reset()
    self.step_impl(action)
    if self.terminal():
      return dm_env.termination(reward=0.0, observation=self.observation())
    else:
      return dm_env.transition(reward=0.0, observation=self.observation())

  def observation_spec(self) -> Any:
    """Returns observation specifications."""
    num_robots = len(self._robots)
    robot_features = {
        'joint_configurations': specs.BoundedArray(
            dtype=np.float32,
            shape=(num_robots, self._dof_by_robot[0]),
            minimum=self._joint_lower_limits[0],
            maximum=self._joint_upper_limits[0],
        ),
        'joint_velocities': specs.BoundedArray(
            dtype=np.float32,
            shape=(num_robots, self._dof_by_robot[0]),
            minimum=-self._config.max_joint_velocity,
            maximum=self._config.max_joint_velocity,
        ),
        'base_poses': specs.BoundedArray(
            dtype=np.float32,
            shape=(num_robots, 4, 4),
            minimum=POSE_BOUNDS_MIN,
            maximum=POSE_BOUNDS_MAX,
        ),
        'tip_relative_poses': specs.BoundedArray(
            dtype=np.float32,
            shape=(num_robots, 4, 4),
            minimum=POSE_BOUNDS_MIN,
            maximum=POSE_BOUNDS_MAX,
        ),
        'robot_index': specs.BoundedArray(
            dtype=np.float32,
            shape=(num_robots, num_robots),
            minimum=0,
            maximum=1,
        ),
    }
    return dict(
        step=specs.BoundedArray(
            dtype=np.float32,
            shape=(),
            minimum=0.0,
            maximum=self._max_steps,
        ),
        robots=robot_features,
    )

  def action_spec(self) -> specs.BoundedArray:
    """Returns action specifications."""

    # We are assuming all robots have the same DoF.
    for i, robot in enumerate(self._robots):
      assert self._dof_by_robot[i] == self._dof_by_robot[0], (
          f'Robots must have the same DoF. {robot} has '
          f'{self._dof_by_robot[i]} and {robot} has '
          f'{self._dof_by_robot[0]}.')

    return specs.BoundedArray(
        shape=(len(self._robots), self._dof_by_robot[0]),
        dtype=np.float32,
        minimum=-self._config.max_joint_velocity,
        maximum=self._config.max_joint_velocity,
    )

  def observation(self) -> Any:
    """Returns the current environment observations."""
    assert self._world is not None
    num_robots = len(self._robots)
    joint_configurations = np.zeros(
        shape=(num_robots, self._dof_by_robot[0]), dtype=np.float32
    )
    joint_velocities = np.zeros(
        shape=(num_robots, self._dof_by_robot[0]), dtype=np.float32
    )
    base_poses = np.zeros(shape=(num_robots, 4, 4), dtype=np.float32)
    tip_relative_poses = np.zeros(shape=(num_robots, 4, 4), dtype=np.float32)
    robot_index = np.eye(num_robots, dtype=np.float32)
    for robot_idx, robot in enumerate(self._robots):
      joint_configurations[robot_idx, :] = self._world.GetJointValues(
          robot
      ).astype(np.float32)
      joint_velocities[robot_idx, :] = self._current_joint_velocities[robot_idx]
      base_pose = self._world.GetRobotBaseTransform(robot)
      base_poses[robot_idx, :, :] = base_pose
      tip_pose = self._world.GetRobotTcpTransform(robot)
      tip_relative_poses[robot_idx, :, :] = relative_pose(base_pose, tip_pose)
    robot_features = {
        'joint_configurations': joint_configurations,
        'joint_velocities': joint_velocities,
        'base_poses': base_poses,
        'tip_relative_poses': tip_relative_poses,
        'robot_index': robot_index,
    }
    return dict(
        step=np.array(self._step, dtype=np.float32),
        robots=robot_features,
    )

  def robot_tip_absolute_poses(self) -> np.ndarray:
    """Returns the current robot tip absolute poses."""
    poses = np.zeros(shape=(len(self._robots), 4, 4), dtype=np.float32)
    for i, robot in enumerate(self._robots):
      poses[i, :, :] = self._world.GetRobotTcpTransform(robot)
    return poses

  def joint_configurations(self) -> np.ndarray:
    """Returns the current joint configurations."""
    joint_configurations = np.zeros(
        shape=(len(self._robots), self._dof_by_robot[0]), dtype=np.float32
    )
    for i, robot in enumerate(self._robots):
      joint_configurations[i, :] = self._world.GetJointValues(robot)
    return joint_configurations

  def set_joint_configurations(self, joint_configurations: np.ndarray):
    """Sets the current joint configurations (assumed to be collision-free)."""
    for i, robot in enumerate(self._robots):
      self._world.SetJointValues(robot, joint_configurations[i, :])

  def randomize_joint_configurations(self):
    """Randomizes the current joint configurations."""
    # For efficiency we move the robots one at a time. We have to random shuffle
    # the order every time so we don't introduce a bias due to the ordering.
    assert not self._config.deterministic_mode
    shuffled_robots = copy.copy(self._robots)
    random.shuffle(shuffled_robots)
    for robot_idx, robot in enumerate(shuffled_robots):
      lower = self._joint_lower_limits[robot_idx]
      upper = self._joint_upper_limits[robot_idx]
      while True:
        random_config = np.zeros_like(lower)
        for i in range(self._dof_by_robot[robot_idx]):
          random_config[i] = np.random.uniform(low=lower[i], high=upper[i])
        self._world.SetJointValues(robot, random_config)
        if not self._robots_in_collision():
          break
      self._world.SetJointValues(robot, self._world.GetJointValues(robot))

  def terminal(self) -> bool:
    """Returns whether the current environment step is terminal."""
    return self._step >= self._max_steps or self._step == RESET_ON_NEXT_STEP
