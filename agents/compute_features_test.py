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

from absl.testing import absltest
import jax
import numpy as np
from scipy.spatial import transform
from agents import compute_features
from environments import base_env
from environments import planning_env


class ComputeFeaturesTest(absltest.TestCase):

  def make_config(self):
    base_config = base_env.BaseEnvConfig(
        world_file_path='mujoco_world/4_pandas_world.xml',
        episode_timeout_s=5.0,
        timestep_s=0.5,
    )
    return planning_env.PlanningEnvConfig(
        base_config=base_config,
        randomize_robot_placement=False,
        num_obstacles_to_generate=2,
        num_targets_to_generate=2,
        obstacles_by_decomposition={},
    )

  def make_env(self, config):
    return planning_env.PlanningEnv(config=config)

  def reconstruct_validate_pose(
      self, pose_features: jax.Array
  ) -> tuple[np.ndarray, np.ndarray]:
    """Returns reconstructed (position, rotation) from a feature vector."""
    translation = np.array(pose_features[0:3] * base_env.WORLD_SPAN_M / 2)
    rotation = np.zeros((3, 3))
    rotation[0, :] = pose_features[3:6]
    rotation[1, :] = pose_features[6:9]
    rotation[2, :] = np.cross(rotation[0, :], rotation[1, :])
    # Validate that the rotation is a valid rotation matrix while we are at it.
    # A rotation matrix is valid if and only if it's orthogonal (transpose
    # equals inverse).
    np.testing.assert_allclose(
        np.linalg.inv(rotation), np.transpose(rotation), atol=1e-5
    )
    return translation, rotation

  def make_pose(
      self, translation: np.ndarray, rotation: np.ndarray
  ) -> np.ndarray:
    """Returns a 4x4 pose matrix combining translation with rotation."""
    pose = np.eye(4)
    pose[:3, 3] = translation
    pose[:3, :3] = rotation
    return pose

  def test_sample_unit_vector(self):
    octants_seen = set()
    for _ in range(1000):
      random_vec = compute_features.sample_unit_vector()
      self.assertAlmostEqual(np.linalg.norm(random_vec), 1.0, places=5)
      signs = tuple((random_vec > 0.0).tolist())
      octants_seen.add(signs)
    self.assertLen(octants_seen, 8)

  def test_add_noise_to_pose(self):
    original_pose = np.eye(4)
    original_pose[:3, 3] = np.array([0.5, 0.6, 0.7])
    original_rotation = transform.Rotation.from_rotvec(
        np.array([1.0, 2.0, 3.0]))
    original_pose[:3, :3] = original_rotation.as_matrix()
    for _ in range(100):
      noisy_pose = compute_features.add_noise_to_pose(
          original_pose=original_pose, max_translation_noise=0.5,
          max_rotation_noise=0.5)
      translation_diff = noisy_pose[:3, 3] - original_pose[:3, 3]
      rotation_diff = (
          original_rotation.inv() * transform.Rotation.from_matrix(
              noisy_pose[:3, :3]))
      translation_noise_mag = np.linalg.norm(translation_diff)
      rotation_noise_mag = rotation_diff.magnitude()
      self.assertGreater(translation_noise_mag, 0.0)
      self.assertGreater(rotation_noise_mag, 0.0)
      self.assertLessEqual(translation_noise_mag, 0.5)
      self.assertLessEqual(rotation_noise_mag, 0.5)
    zero_noise_added = compute_features.add_noise_to_pose(
        original_pose=original_pose, max_translation_noise=0.0,
        max_rotation_noise=0.0)
    np.testing.assert_allclose(zero_noise_added, original_pose)

  def test_box_features(self):
    spans = np.array([1.0, 2.0, 3.0])
    box_pose = np.eye(4)
    features = compute_features.calculate_box_features(spans, box_pose)
    np.testing.assert_allclose(features[:, 3], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(features[:, 0], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(features[:, 1], [0.0, 2.0, 0.0])
    np.testing.assert_allclose(features[:, 2], [0.0, 0.0, 3.0])

    # Translation.
    box_pose[:3, 3] = np.array([0.5, 0.6, 0.7])
    features = compute_features.calculate_box_features(spans, box_pose)
    np.testing.assert_allclose(features[:, 3], [0.5, 0.6, 0.7])
    np.testing.assert_allclose(features[:, 0], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(features[:, 1], [0.0, 2.0, 0.0])
    np.testing.assert_allclose(features[:, 2], [0.0, 0.0, 3.0])

    # And rotation around x axis.
    box_pose[:3, :3] = np.array([[1.0, 0.0, 0.0],
                                 [0.0, 0.0, -1.0],
                                 [0.0, 1.0, 0.0]])
    features = compute_features.calculate_box_features(spans, box_pose)
    np.testing.assert_allclose(features[:, 3], [0.5, 0.6, 0.7])
    np.testing.assert_allclose(features[:, 0], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(features[:, 1], [0.0, 0.0, 2.0])
    np.testing.assert_allclose(features[:, 2], [0.0, -3.0, 0.0])

  def test_compute_features(self):
    env_config = self.make_config()
    env = self.make_env(env_config)
    env.reset()

    action = np.zeros((4, 7))
    action[0, :] = np.array([0.02] * 7)
    action[1, :] = np.array([0.02] * 7)
    env.step(action)

    obs = env.observation()
    obs_spec = env.observation_spec()
    graphs_tuple = compute_features.make_graph_features(
        obs,
        obs_spec,
        compute_features.FeatureConfig(
            robot_base_features=True,
            robot_relative_tip_features=True,
            max_translation_noise=0.0,
            max_rotation_noise=0.0,
        ),
    )

    # 4 arms + 2 targets + 2 obstacles = 8 nodes.
    self.assertSequenceEqual(graphs_tuple.n_node.tolist(), (8,))

    # 4 arm x (2 targets + 3 arms + 2 obstacles) = 28 edges.
    self.assertSequenceEqual(graphs_tuple.n_edge.tolist(), (28,))

    # Arm-arm edges (edges 0 to 11).
    self.assertSequenceEqual(
        graphs_tuple.senders[0:12].tolist(),
        [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2],
    )
    self.assertSequenceEqual(
        graphs_tuple.receivers[0:12].tolist(),
        [0] * 3 + [1] * 3 + [2] * 3 + [3] * 3,
    )

    # Target-arm edges (edges 12 to 20).
    self.assertSequenceEqual(
        graphs_tuple.senders[12:20].tolist(),
        [4, 5] * 4,
    )
    self.assertSequenceEqual(
        graphs_tuple.receivers[12:20].tolist(),
        [0] * 2 + [1] * 2 + [2] * 2 + [3] * 2,
    )

    # Obstacle-arm edges (edges 20 to 28).
    self.assertSequenceEqual(
        graphs_tuple.senders[20:28].tolist(),
        [6, 7] * 4,
    )
    self.assertSequenceEqual(
        graphs_tuple.receivers[20:28].tolist(),
        [0] * 2 + [1] * 2 + [2] * 2 + [3] * 2,
    )

    nodes = graphs_tuple.nodes
    self.assertIsInstance(nodes, jax.Array)
    edges = graphs_tuple.edges
    self.assertIsInstance(edges, jax.Array)
    globals_ = graphs_tuple.globals
    self.assertIsInstance(globals_, jax.Array)

    # All features should be normalised.
    self.assertTrue((np.array(nodes) <= 1.0).all())
    self.assertTrue((np.array(nodes) >= -1.0).all())
    self.assertTrue((np.array(edges) <= 1.0).all())
    self.assertTrue((np.array(edges) >= -1.0).all())
    self.assertTrue((np.array(globals_) <= 1.0).all())
    self.assertTrue((np.array(globals_) >= -1.0).all())

    def normalize_joint_configs(joint_config: np.ndarray) -> np.ndarray:
      mins = env._world.GetLowerJointPositionLimits('panda1')
      ranges = env._world.GetUpperJointPositionLimits(
          'panda1'
      ) - env._world.GetLowerJointPositionLimits('panda1')
      return 2.0 * (joint_config - mins) / ranges - 1.0

    feature_idx = 0

    # Arm features.
    joint_configurations = nodes[0:4, feature_idx : feature_idx + 7]
    self.assertSequenceAlmostEqual(
        joint_configurations[0, :].tolist(),
        normalize_joint_configs(
            np.array([0.01, -0.7754, 0.01, -2.1717, 0.01, 1.4062, 0.01])
        ),
        places=3,
    )
    self.assertSequenceAlmostEqual(
        joint_configurations[2, :].tolist(),
        normalize_joint_configs(
            np.array([0.0, -0.7854, 0.0, -2.1817, 0.0, 1.3962, 0.0])
        ),
        places=3,
    )
    feature_idx += 7

    joint_velocities = nodes[0:4, feature_idx : feature_idx + 7]
    self.assertSequenceAlmostEqual(
        joint_velocities[0, :].tolist(),
        np.array(([0.02 / 2.175] * 7)),
        places=3,
    )
    self.assertSequenceAlmostEqual(
        joint_velocities[2, :].tolist(),
        np.array(([0.0] * 7)),
        places=3,
    )
    feature_idx += 7

    dwelling = nodes[0:4, feature_idx : feature_idx + 1]
    np.testing.assert_allclose(dwelling, -1.0 * np.ones((4, 1)))
    feature_idx += 1

    base_poses = nodes[0:4, feature_idx : feature_idx + 9]
    robot_0_translation, robot_0_rotation = self.reconstruct_validate_pose(
        base_poses[0, :]
    )
    robot_2_translation, robot_2_rotation = self.reconstruct_validate_pose(
        base_poses[2, :]
    )
    np.testing.assert_allclose(
        robot_0_translation,
        np.array([0.5515, -0.6285, 0.009]),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        robot_2_translation,
        np.array([-0.5515, 0.6285, 0.009]),
        atol=1e-5,
    )

    np.testing.assert_allclose(
        robot_0_rotation,
        # Rz(pi)
        np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        robot_2_rotation,
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        atol=1e-5,
    )
    feature_idx += 9

    tip_relative_poses = nodes[0:4, feature_idx : feature_idx + 9]
    tip0_relative_position, tip0_relative_rotation = (
        self.reconstruct_validate_pose(tip_relative_poses[0, :])
    )
    tip2_relative_position, _ = self.reconstruct_validate_pose(
        tip_relative_poses[2, :]
    )
    np.testing.assert_allclose(
        tip0_relative_position,
        np.array([0.24346292018, 0.0084692, 0.51231]),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        tip2_relative_position,
        np.array([0.23822, 0.0, 0.510281]),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        tip0_relative_rotation,
        np.array([
            [-0.772286, -0.008976, -0.635212],
            [-0.008807, 0.999955, -0.003422],
            [0.635214, 0.002951, -0.77233],
        ]),
        atol=1e-5,
    )
    feature_idx += 9

    # Target features.
    target_done = nodes[4:6, feature_idx : feature_idx + 1]
    self.assertSequenceAlmostEqual(target_done[0, :].tolist(), [-1.0])
    feature_idx += 1

    # Obstacle features.
    obstacle_markers = nodes[6:8, feature_idx : feature_idx + 1]
    np.testing.assert_allclose(obstacle_markers, 1.0)
    feature_idx += 1

    self.assertEqual(8, nodes.shape[0])
    self.assertEqual(feature_idx, nodes.shape[1])

    # Edge features.
    feature_idx = 0
    arm_arm_rel_poses = edges[0:12, feature_idx : feature_idx + 9]

    # Arm-Arm Edge 4 (arm 2 to arm 1).
    edge4_rel_position, edge4_rel_rotation = self.reconstruct_validate_pose(
        arm_arm_rel_poses[4, :]
    )

    np.testing.assert_allclose(
        edge4_rel_position,
        np.array([0.0, 1.257, 0.0]),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        edge4_rel_rotation,
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        atol=1e-5,
    )
    feature_idx += 9

    arm_target_rel_poses = edges[12:20, feature_idx : feature_idx + 9]

    # Target-Arm Edge 3 (target 1 to arm 1).
    relative_pose = self.make_pose(
        *self.reconstruct_validate_pose(arm_target_rel_poses[3, :])
    )

    target_pose = obs['targets']['poses'][1, :, :]
    tip_pose = np.matmul(
        obs['robots']['base_poses'][1, :, :],
        obs['robots']['tip_relative_poses'][1, :, :],
    )
    expected_relative_pose = np.matmul(np.linalg.inv(tip_pose), target_pose)

    np.testing.assert_allclose(relative_pose, expected_relative_pose, atol=1e-5)
    feature_idx += 9

    # Obstacle-Arm Edge 3 (obstacle 1 to arm 1). This is more extensively tested
    # in test_box_features().
    arm_obs_features = edges[20:28, feature_idx : feature_idx + 12]
    base_pose_scaled = obs['robots']['base_poses'][1, :, :]
    base_pose_scaled[:3, 3] /= base_env.WORLD_SPAN_M / 2
    tip_relative_pose_scaled = obs['robots']['tip_relative_poses'][1, :, :]
    tip_relative_pose_scaled[:3, 3] /= base_env.WORLD_SPAN_M / 2
    tip_pose = np.matmul(
        base_pose_scaled,
        tip_relative_pose_scaled,
    )

    # Obstacle bases in tip frame.
    obstacle_relative_bases = arm_obs_features[3].reshape((3, 4))
    # The last column should be the centroid, relative to tip.
    relative_centroid = np.array(obstacle_relative_bases[:, 3].tolist() + [1.0])
    centroid = np.matmul(tip_pose, relative_centroid)[:3]
    expected_centroid = obs['obstacles']['poses'][1, :3, 3] / (
        base_env.WORLD_SPAN_M / 2)

    np.testing.assert_allclose(centroid, expected_centroid, atol=1e-5)
    feature_idx += 12

    self.assertEqual(feature_idx, edges.shape[1])

    # Globals.
    self.assertAlmostEqual(globals_[0][0], (1.0 / 10.0) * 2.0 - 1.0)

  def test_compute_features_absolute(self):
    env = self.make_env(self.make_config())
    env.reset()

    action = np.zeros((4, 7))
    action[0, :] = np.array([0.02] * 7)
    action[1, :] = np.array([0.02] * 7)
    env.step(action)

    obs = env.observation()
    obs_spec = env.observation_spec()
    # Here we set robot_base_features to False to test that it gets generated
    # anyways if we have use_relative_features=False.
    graphs_tuple = compute_features.make_graph_features(
        obs, obs_spec, compute_features.FeatureConfig(
            robot_base_features=False, robot_relative_tip_features=False,
            max_translation_noise=0.0, max_rotation_noise=0.0,
            use_relative_features=False))

    # 4 arms + 2 targets + 2 obstacles = 8 nodes.
    self.assertSequenceEqual(graphs_tuple.n_node.tolist(), (8,))

    # 4 arm x (2 targets + 3 arms + 2 obstacles) = 28 edges.
    self.assertSequenceEqual(graphs_tuple.n_edge.tolist(), (28,))

    # Arm-arm edges (edges 0 to 11).
    self.assertSequenceEqual(
        graphs_tuple.senders[0:12].tolist(),
        [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2],
    )
    self.assertSequenceEqual(
        graphs_tuple.receivers[0:12].tolist(),
        [0] * 3 + [1] * 3 + [2] * 3 + [3] * 3,
    )

    # Target-arm edges (edges 12 to 20).
    self.assertSequenceEqual(
        graphs_tuple.senders[12:20].tolist(),
        [4, 5] * 4,
    )
    self.assertSequenceEqual(
        graphs_tuple.receivers[12:20].tolist(),
        [0] * 2 + [1] * 2 + [2] * 2 + [3] * 2,
    )

    # Obstacle-arm edges (edges 20 to 28).
    self.assertSequenceEqual(
        graphs_tuple.senders[20:28].tolist(),
        [6, 7] * 4,
    )
    self.assertSequenceEqual(
        graphs_tuple.receivers[20:28].tolist(),
        [0] * 2 + [1] * 2 + [2] * 2 + [3] * 2,
    )

    nodes = graphs_tuple.nodes
    self.assertIsInstance(nodes, jax.Array)

    # All features should be normalised.
    self.assertTrue((np.array(nodes) <= 1.0).all())
    self.assertTrue((np.array(nodes) >= -1.0).all())

    feature_idx = 0

    # Arm features.
    # Joint configurations.
    feature_idx += 7

    # Joint velocities.
    feature_idx += 7

    dwelling = nodes[0:4, feature_idx : feature_idx + 1]
    np.testing.assert_allclose(dwelling, -1.0 * np.ones((4, 1)))
    feature_idx += 1

    base_poses = nodes[0:4, feature_idx : feature_idx + 9]
    robot_0_translation, robot_0_rotation = self.reconstruct_validate_pose(
        base_poses[0, :]
    )
    np.testing.assert_allclose(
        robot_0_translation,
        np.array([0.5515, -0.6285, 0.009]),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        robot_0_rotation,
        # Rz(pi)
        np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]),
        atol=1e-5,
    )
    feature_idx += 9

    # Target features.
    target_done = nodes[4:6, feature_idx : feature_idx + 1]
    self.assertSequenceAlmostEqual(target_done[0, :].tolist(), [-1.0])
    feature_idx += 1

    target_poses = nodes[4:6, feature_idx : feature_idx + 9]
    target0_pose = self.make_pose(
        *self.reconstruct_validate_pose(target_poses[0, :])
    )
    np.testing.assert_allclose(
        target0_pose,
        obs['targets']['poses'][0, :, :],
        atol=1e-5,
    )
    feature_idx += 9

    # Obstacle features.
    obstacle_spans = nodes[6:8, feature_idx : feature_idx + 3]
    np.testing.assert_allclose(obstacle_spans[0] * 5.0,
                               obs['obstacles']['spans'][0], atol=1e-5)
    feature_idx += 3

    obstacle_poses = nodes[6:8, feature_idx : feature_idx + 9]
    obstacle0_pose = self.make_pose(
        *self.reconstruct_validate_pose(obstacle_poses[0, :])
    )
    np.testing.assert_allclose(
        obstacle0_pose,
        obs['obstacles']['poses'][0, :, :],
        atol=1e-5,
    )
    feature_idx += 9

    self.assertEqual(8, nodes.shape[0])
    self.assertEqual(feature_idx, nodes.shape[1])


if __name__ == '__main__':
  absltest.main()
