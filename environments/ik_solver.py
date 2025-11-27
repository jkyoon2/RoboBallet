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

"""IK solver for robot arms."""

import datetime
from typing import Sequence

import mujoco
import numpy as np


class IkSolver:
  """Inverse kinematics solver for robot arms.
  """

  def __init__(
      self,
      model: mujoco.MjModel,
      controllable_joints_ids: list[int],
      site_name: str,
      collision_pairs: Sequence[tuple[Sequence[str], Sequence[str]]] | None,
      solver_configs: dict[str, float | bool] | None = None,
  ):
    """Constructor.

    Args:
      model: The MuJoCo model.
      controllable_joints_ids: List of joints to be controlled. Only 1 DoF
        joints are supported.
      site_name: Name of the MuJoCo site to control, i.e. the frame to be
        controlled.
      collision_pairs: Optional collision pairs to avoid (geom names).
      solver_configs: Solver config options.
    """
    raise NotImplementedError('IK solver not implemented.')

  def solve(
      self,
      desired_global_pos_tcp: np.ndarray,
      desired_global_quat_tcp: np.ndarray,
      initial_joint_configuration: np.ndarray,
      linear_tol: float = 1e-3,
      angular_tol: float = 1e-3,
      max_steps: int = 100,
      early_stop: bool = False,
  ) -> tuple[np.ndarray | None, list[np.ndarray]]:
    """Solve the inverse kinematics to a desired pose.

    Args:
      desired_global_pos_tcp: Target position of the site w.r.t. the world
        frame.
      desired_global_quat_tcp: Target orientation of the site w.r.t. the world
        frame represented as a quaternion ([w,x,y,z]).
      initial_joint_configuration: Joint configuration to start from.
      linear_tol: The linear tolerance, in meters, that determines if the
        solution found is valid.
      angular_tol: The angular tolerance, in radians, to determine if the
        solution found is valid.
      max_steps: Maximum number of integration steps.
      early_stop: If true, stops the attempt as soon as the configuration is
        within the linear and angular tolerances. If false, it will always run
        `max_steps` iterations and return the last configuration.

    Returns:
      If a joint configuration is found that satisfies the `linear_tol` and
      `angular_tol` provided, returns the corresponding joint configuration and
      the path of joint configurations.
      Otherwise, returns `None` as the joint configuration and the path found by
      the algorithm.

    Raises:
      ValueError: If the `initial_joint_configuration` does not have the correct
        length.
    """
    if len(initial_joint_configuration) != len(self._controllable_joints_ids):
      raise ValueError(
          'The provided initial joint configuration does not have the right'
          ' number of elements expected length of'
          f' {len(self._controllable_joints_ids)}. Got'
          f' {initial_joint_configuration}'
      )

    raise NotImplementedError('IK solver not implemented.')
