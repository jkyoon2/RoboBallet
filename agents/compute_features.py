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

"""Library for computing GNN features from the environment observations."""

from collections.abc import Iterable, Mapping
from typing import Any

import chex
import dm_env
from flax import struct
import jax
import jax.numpy as jnp
import jraph
import numpy as np
import scipy
from scipy.spatial import transform


ArraySpecTree = (
    dm_env.specs.BoundedArray
    | Iterable['ArraySpecTree']
    | Mapping[str, 'ArraySpecTree']
)


class FeatureConfig(struct.PyTreeNode):
  """Network hyper-parameters.

  Attributes:
    robot_relative_tip_features: Use relative tip features.
    robot_base_features: Use base position features.
    relative_poses_use_tip: Relative poses (edge->robot and obstacle->robot) are
      relative to robot tip instead of robot base.
    use_relative_features: Encode delta poses as edge features instead of
      absolute poses as node features. Set to false for non-graphnet models (eg
      transformers).
    use_robot_id_features: Use robot ID features.
    max_translational_noise: Maximum translational noise for obstacles in m.
    max_rotation_noise: Maximum rotational noise for obstacles in radians.
  """

  robot_relative_tip_features: bool = False
  robot_base_features: bool = False
  relative_poses_use_tip: bool = True
  use_relative_features: bool = True
  use_robot_id_features: bool = False

  max_translation_noise: float = 0.0
  max_rotation_noise: float = 0.0


def scale_array(
    array: np.ndarray, array_spec: dm_env.specs.BoundedArray
) -> np.ndarray:
  """Scales a single array to [-1, 1] based on range in spec."""
  if array.dtype == np.bool_ or array.dtype == bool:
    # Scale boolean arrays to [-1, 1].
    return array.astype(np.float32) * 2.0 - 1.0
  elif array.dtype == np.float32:
    value_range = array_spec.maximum - array_spec.minimum
    return 2.0 * (array - array_spec.minimum) / value_range - 1.0
  else:
    raise ValueError(f'Unsupported dtype: {array.dtype}')


def _make_pose_features(pose: np.ndarray) -> np.ndarray:
  """Calculates pose features for the network.

  Converts a 4x4 pose matrix to the concatenation of:

  1. 3D translation.

  2. 6D rotational representation described in -
  Zhou, Yi, et al. "On the continuity of rotation representations in neural
  networks." Proceedings of the IEEE/CVF Conference on Computer Vision and
  Pattern Recognition. 2019.

  Args:
    pose: pose matrix (4x4). May be single or batched (Bx4x4).

  Returns:
    vec9d: 9D representation of the pose.
  """
  if len(pose.shape) == 3:
    translation_part = pose[:, :3, 3].reshape(-1, 3)
    # The representation is the flattened top two rows of the rotation matrix.
    rotation_part = pose[:, :2, :3].reshape(-1, 6)
    return np.concatenate([translation_part, rotation_part], axis=1)
  else:
    translation_part = pose[:3, 3].reshape((3,))
    rotation_part = pose[:2, :3].reshape((6,))
    return np.concatenate([translation_part, rotation_part], axis=0)


def sample_unit_vector() -> np.ndarray:
  """Samples a unit vector uniformly distributed on unit sphere."""
  # Marsaglia (1972) http://doi.org/10.1214/aoms/1177692644
  # Explanation: https://mathworld.wolfram.com/SpherePointPicking.html
  x1 = None
  x2 = None
  while x1 is None or ((x1 * x1 + x2 * x2) >= 1.0):
    x1 = np.random.uniform(low=-1.0, high=1.0)
    x2 = np.random.uniform(low=-1.0, high=1.0)
  s = x1 * x1 + x2 * x2
  return np.array([
      2.0 * x1 * np.sqrt(1.0 - s),
      2.0 * x2 * np.sqrt(1.0 - s),
      1.0 - 2.0 * s
  ], dtype=np.float32)


def add_noise_to_pose(original_pose: np.ndarray, max_translation_noise: float,
                      max_rotation_noise: float) -> np.ndarray:
  """Adds a random rotation and translation to the pose."""
  translation_mag = np.random.uniform(low=0.0, high=max_translation_noise)
  random_translation = sample_unit_vector() * translation_mag
  rotation_mag = np.random.uniform(low=0.0, high=max_rotation_noise)
  rotation_vector = sample_unit_vector() * rotation_mag
  random_rotation = transform.Rotation.from_rotvec(rotation_vector)
  new_rotation = np.matmul(random_rotation.as_matrix(), original_pose[:3, :3])
  new_translation = random_translation + original_pose[:3, 3]
  new_pose = np.eye(4, dtype=np.float32)
  new_pose[:3, :3] = new_rotation
  new_pose[:3, 3] = new_translation
  return new_pose


def calculate_box_features(spans: np.ndarray,
                           box_pose: np.ndarray) -> np.ndarray:
  """Calculates centroid + scaled basis vectors that define each box."""
  # Centroid is the translation of the pose, and scaled basis vectors are the
  # column vectors of the rotation part of the pose, scaled by spans. We can
  # de-homogenize by removing the last row.

  # Pose is:
  # [     ] [ | ]
  # [  R  ] [ T ]
  # [     ] [ | ]
  # 0  0  0   1
  #
  # Where R is the rotation (each column is a basis vector), and T is the
  # translation (centroid).

  # Start by dropping the last row.
  ret = box_pose[:3, :].copy()

  # Then multiply each basis vector by the corresponding dim in spans.
  for dim in range(3):
    ret[:, dim] *= spans[dim]

  # Now the first 3 columns are scaled basis vectors relative to centroid, and
  # the last column is the centroid.
  return ret


def make_graph_features_prescaled(
    observation: Any, feature_config: FeatureConfig
) -> jraph.GraphsTuple:
  """Creates GNN graph data from observations and corresponding specs."""
  robots = observation['robots']
  robot_features = [
      robots['joint_configurations'],
      robots['joint_velocities'],
      np.reshape(robots['dwelling'], (-1, 1)),
  ]

  # If we are not using relative features, we need to add the base poses as
  # node features, otherwise there would be no way for the network to figure out
  # where the robots are.
  if (feature_config.robot_base_features or
      not feature_config.use_relative_features):
    robot_features.append(_make_pose_features(robots['base_poses']))

  # This is tip pose relative to base pose. Technically redundant as the network
  # can calculate it from the base pose and joint configuration.
  if feature_config.robot_relative_tip_features:
    robot_features.append(_make_pose_features(robots['tip_relative_poses']))

  # Add robot ID features.
  if feature_config.use_robot_id_features:
    robot_features.append(robots['robot_index'])

  robot_nodes = np.concatenate(robot_features, axis=1)
  num_robots = robot_nodes.shape[0]

  def _robot_node_idx(robot_idx):
    return robot_idx

  targets = observation['targets']
  target_features = [
      targets['done'].reshape((targets['done'].shape[0], 1)).astype(np.float32)
  ]
  if not feature_config.use_relative_features:
    target_features.append(_make_pose_features(targets['poses']))
  target_nodes = np.concatenate(target_features, axis=1)
  num_targets = target_nodes.shape[0]

  def _target_node_idx(target_idx):
    return target_idx + num_robots

  obstacles = observation['obstacles']

  num_obstacles = obstacles['spans'].shape[0]
  if feature_config.use_relative_features:
    # All obstacle features are edge, so here we just put a ones() marker to
    # denote that it's an obstacle node.
    obstacle_nodes = np.ones((num_obstacles, 1), dtype=np.float32)
  else:
    obstacle_nodes = np.concatenate([
        obstacles['spans'],
        _make_pose_features(obstacles['poses']),
    ], axis=1)

  def _obstacle_node_idx(obstacle_idx):
    return obstacle_idx + num_robots + num_targets

  edge_senders = []
  edge_receivers = []

  base_poses = robots['base_poses']
  base_pose_inverses = [np.linalg.inv(pose) for pose in base_poses]
  tip_poses = np.matmul(base_poses, robots['tip_relative_poses'])
  tip_pose_inverses = [np.linalg.inv(pose) for pose in tip_poses]

  # Robot to robot edges.
  num_robot_to_robot_edges = num_robots * (num_robots - 1)
  relative_base_pose_features = np.zeros(
      shape=(num_robot_to_robot_edges, 9), dtype=np.float32
  )
  edge_index = 0
  for receiving_robot in range(num_robots):
    for sending_robot in range(num_robots):
      if receiving_robot == sending_robot:
        continue
      sender_pose = base_poses[sending_robot, :, :]
      relative_pose = np.matmul(
          base_pose_inverses[receiving_robot], sender_pose
      )
      relative_base_pose_features[edge_index, :] = _make_pose_features(
          relative_pose
      )
      edge_senders.append(_robot_node_idx(sending_robot))
      edge_receivers.append(_robot_node_idx(receiving_robot))
      edge_index += 1
  robot_robot_features = [relative_base_pose_features]
  robot_t_robot_edges = np.concatenate(robot_robot_features, axis=1)

  robot_pose_inverses = (
      tip_pose_inverses if feature_config.relative_poses_use_tip
      else base_pose_inverses)

  # Target to robot edges.
  num_target_to_robot_edges = num_targets * num_robots
  relative_poses = np.zeros(
      shape=(num_target_to_robot_edges, 9), dtype=np.float32
  )
  target_poses = targets['poses']

  for receiving_robot in range(num_robots):
    for sending_target in range(num_targets):
      edge_index = receiving_robot * num_targets + sending_target
      relative_pose = np.matmul(
          robot_pose_inverses[receiving_robot], target_poses[sending_target]
      )
      relative_poses[edge_index, :] = _make_pose_features(relative_pose)
      edge_senders.append(_target_node_idx(sending_target))
      edge_receivers.append(_robot_node_idx(receiving_robot))
  target_t_robot_edges = relative_poses

  # Obstacle to robot edges.
  num_obstacle_to_robot_edges = num_obstacles * num_robots
  bases_features = np.zeros(
      # Each obstacle has basis (3 vectors) + centroid
      shape=(num_obstacle_to_robot_edges, 12), dtype=np.float32
  )
  obstacle_poses = obstacles['poses']
  noisy_obstacle_poses = [
      add_noise_to_pose(
          original_pose=pose,
          max_translation_noise=feature_config.max_translation_noise,
          max_rotation_noise=feature_config.max_rotation_noise)
      for pose in obstacle_poses]
  for receiving_robot in range(num_robots):
    for sending_obstacle in range(num_obstacles):
      edge_index = receiving_robot * num_obstacles + sending_obstacle
      relative_pose = np.matmul(
          robot_pose_inverses[receiving_robot],
          noisy_obstacle_poses[sending_obstacle]
      )
      spans = obstacles['spans'][sending_obstacle]
      box_features = calculate_box_features(spans=spans, box_pose=relative_pose)
      bases_features[edge_index, :] = box_features.flatten()
      edge_senders.append(_obstacle_node_idx(sending_obstacle))
      edge_receivers.append(_robot_node_idx(receiving_robot))
  obstacle_robot_features = [bases_features]
  obstacle_t_robot_edges = np.concatenate(obstacle_robot_features, axis=1)

  # Global features.
  episode_time = observation['step'].astype(np.float32)
  current_score = observation['current_score'].astype(np.float32)
  global_features = jnp.concatenate([
      episode_time.reshape((1,)),
      current_score.reshape((1,)),
  ])

  nodes = jnp.array(
      scipy.linalg.block_diag(robot_nodes, target_nodes, obstacle_nodes)
  )
  edges = jnp.array(
      scipy.linalg.block_diag(
          robot_t_robot_edges, target_t_robot_edges, obstacle_t_robot_edges
      )
  )
  return jraph.GraphsTuple(
      nodes=nodes,
      edges=edges,
      receivers=jnp.array(edge_receivers),
      senders=jnp.array(edge_senders),
      globals=jnp.expand_dims(global_features, axis=0),
      n_node=jnp.array([nodes.shape[0]], dtype=np.uint32),
      n_edge=jnp.array([edges.shape[0]], dtype=np.uint32),
  )


def make_graph_features(
    observation: chex.ArrayTree,
    observation_spec: ArraySpecTree,
    feature_config: FeatureConfig,
) -> jraph.GraphsTuple:
  """Creates GNN graph data from observations and corresponding specs."""
  observation = jax.tree_util.tree_map(
      scale_array, observation, observation_spec
  )
  return make_graph_features_prescaled(observation, feature_config)
