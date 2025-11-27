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

"""Trajectory library."""

from collections.abc import Mapping
import dataclasses

import numpy as np

from environments import planning_env
from environments import target as target_lib


ArrayTree = Mapping[str, 'ArrayTree | np.ndarray']


@dataclasses.dataclass(frozen=True)
class Trajectory:
  """Episode trajectory.

  Attributes:
    start_state: Starting state of the episode.
    observations: Observations at each state.
    actions: Actions taken at each state.
    rewards: Rewards from taking the corresponding action at each state.
    target_states: Target states at each state.
  """

  start_state: planning_env.StartState
  observations: list[ArrayTree]
  actions: list[np.ndarray]  # (n_robots, n_dof)
  rewards: list[float]
  target_states: list[list[target_lib.TargetState]]
