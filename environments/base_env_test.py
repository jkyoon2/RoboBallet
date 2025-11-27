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
from dm_env import specs
from dm_env import test_utils
import numpy as np
from environments import base_env
from environments import geometry


# The EnvironmentTestMixin adds tests for compliance with the dm_env
# semantic contract.
class BaseEnvMixinMujocoTest(
    test_utils.EnvironmentTestMixin, absltest.TestCase
):

  def make_object_under_test(self, timestep=0.5):
    world_file_path = 'mujoco_world/4_pandas_world.xml'
    config = base_env.BaseEnvConfig(
        world_file_path=world_file_path,
        episode_timeout_s=timestep * 10,
        timestep_s=timestep,
    )
    return base_env.BaseEnv(config=config)


class BaseEnvTest(absltest.TestCase):

  def make_simulator(self, timestep=0.5):
    world_file_path = 'mujoco_world/4_pandas_world.xml'
    config = base_env.BaseEnvConfig(
        world_file_path=world_file_path,
        episode_timeout_s=timestep * 10,
        timestep_s=timestep,
    )
    return base_env.BaseEnv(config=config)

  def test_observation_specs(self):
    env = self.make_simulator()

    expected_observation_spec = {
        'step': specs.BoundedArray(
            dtype=np.float32,
            shape=(),
            minimum=0.0,
            maximum=10,
        ),
        'robots': {
            'joint_configurations': specs.BoundedArray(
                dtype=np.float32,
                shape=(4, 7),
                minimum=[
                    -2.8973,
                    -1.7628,
                    -2.8973,
                    -3.0718,
                    -2.8973,
                    -0.0175,
                    -2.8973,
                ],
                maximum=[
                    2.8973,
                    1.7628,
                    2.8973,
                    -0.0698,
                    2.8973,
                    3.7525,
                    2.8973,
                ],
            ),
            'joint_velocities': specs.BoundedArray(
                dtype=np.float32,
                shape=(4, 7),
                minimum=-2.175,
                maximum=2.175,
            ),
            'base_poses': specs.BoundedArray(
                dtype=np.float32,
                shape=(4, 4, 4),
                minimum=base_env.POSE_BOUNDS_MIN,
                maximum=base_env.POSE_BOUNDS_MAX,
            ),
            'tip_relative_poses': specs.BoundedArray(
                dtype=np.float32,
                shape=(4, 4, 4),
                minimum=base_env.POSE_BOUNDS_MIN,
                maximum=base_env.POSE_BOUNDS_MAX,
            ),
            'robot_index': specs.BoundedArray(
                dtype=np.float32,
                shape=(4, 4),
                minimum=0,
                maximum=1,
            ),
        },
    }

    self.assertEqual(env.observation_spec(), expected_observation_spec)

  def test_stepping_max_steps(self):
    env = self.make_simulator()
    action = np.zeros(shape=(4, 7), dtype=np.float32)
    self.assertTrue(env.step(action).first())
    self.assertTrue(env.step(action).mid())
    self.assertTrue(env.step(action).mid())
    self.assertTrue(env.step(action).mid())
    self.assertTrue(env.step(action).mid())
    self.assertTrue(env.step(action).mid())
    self.assertTrue(env.step(action).mid())
    self.assertTrue(env.step(action).mid())
    self.assertTrue(env.step(action).mid())
    self.assertTrue(env.step(action).mid())
    self.assertTrue(env.step(action).last())

  def test_observations_initial_joint_values(self):
    env = self.make_simulator()
    reset_step = env.reset()
    observation = reset_step.observation
    self.assertEqual(observation['step'], 0)
    self.assertSequenceAlmostEqual(
        observation['robots']['joint_configurations'][0, :],
        [0.0, -0.7854, 0.0, -2.1817, 0.0, 1.3962, 0.0],
        places=3,
    )

  def test_normal_stepping(self):
    env = self.make_simulator()
    env.reset()
    action = np.zeros(shape=(4, 7), dtype=np.float32)
    action[0, :] = np.array([0.02] * 7)
    action[1, :] = np.array([0.02] * 7)
    new_timestep = env.step(action)
    self.assertSequenceAlmostEqual(
        new_timestep.observation['robots']['joint_configurations'][0, :],
        [0.01, -0.7754, 0.01, -2.1717, 0.01, 1.4062, 0.01],
        places=3,
    )
    self.assertSequenceAlmostEqual(
        new_timestep.observation['robots']['joint_configurations'][1, :],
        [0.01, -0.7754, 0.01, -2.1717, 0.01, 1.4062, 0.01],
        places=3,
    )
    self.assertSequenceAlmostEqual(
        new_timestep.observation['robots']['joint_configurations'][2, :],
        [0.0, -0.7854, 0.0, -2.1817, 0.0, 1.3962, 0.0],
        places=3,
    )

  def test_stepping_crashing(self):
    env = self.make_simulator()
    _ = env.reset()
    action = np.zeros(shape=(4, 7), dtype=np.float32)
    action[0, :] = np.array([0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0])
    # Crash panda1 into the table and keep going. We should stop at
    # qpos[1] == 0.4146
    num_robots_in_collision = []
    for _ in range(10):
      new_timestep = env.step(action)
      num_robots_in_collision.append(len(env._robots_in_collision_last_step))
    self.assertAlmostEqual(
        new_timestep.observation['robots']['joint_configurations'][0, 1],
        0.4146,
        places=3,
    )
    self.assertSequenceEqual(
        num_robots_in_collision, [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    )

  def test_collision_recovery(self):
    env = self.make_simulator()
    for _ in range(10):
      env.reset()
      action = (np.random.random(size=(4 * 7)) > 0.5).astype(np.float32) - 0.5
      action = np.reshape(action, (4, 7))
      for _ in range(10):
        _ = env.step(action)

  def test_stepping_acceleration_clipping(self):
    env = self.make_simulator(timestep=0.01)
    # With acceleration limit at 8 rad/s2, we can accelerate by 0.08 rad/s per
    # timestep.
    env.reset()
    action = np.zeros(shape=(4, 7), dtype=np.float32)
    action[0, :] = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # The velocity here should have been clipped to 0.08 rad/s.
    new_timestep = env.step(action)
    self.assertAlmostEqual(
        new_timestep.observation['robots']['joint_configurations'][0, 1],
        -0.7854 + 0.08 * 0.01,
        places=3,
    )

    # Now we request 0.12 and it shouldn't be clipped.
    action[0, 1] = 0.12
    new_timestep = env.step(action)
    self.assertAlmostEqual(
        new_timestep.observation['robots']['joint_configurations'][0, 1],
        -0.7854 + (0.08 + 0.12) * 0.01,
        places=3,
    )

  def test_relative_tip_poses(self):
    # At the start position all relative tip poses should be the same.
    env = self.make_simulator()
    new_timestep = env.reset()
    obs = new_timestep.observation['robots']['tip_relative_poses'][0]
    np.testing.assert_allclose(
        new_timestep.observation['robots']['tip_relative_poses'][1],
        obs,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        new_timestep.observation['robots']['tip_relative_poses'][2],
        obs,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        new_timestep.observation['robots']['tip_relative_poses'][3],
        obs,
        atol=1e-5,
    )

  def test_robot_index(self):
    env = self.make_simulator()
    new_timestep = env.reset()
    obs = new_timestep.observation['robots']
    np.testing.assert_allclose(obs['robot_index'], np.eye(4))

  def test_ik(self):
    env = self.make_simulator()
    env.reset()
    world = env._world
    new_boxes = [world.AddBox(np.array([0.1, 1.0, 0.1])) for _ in range(2)]
    assert not world.RobotsInCollision()
    for robot_name in ['panda1', 'panda3']:
      robot_base_pos = world.GetWorldTransform(robot_name)[:3, 3]

      box_transform = np.eye(4)
      box_transform[:3, 3] = robot_base_pos + np.array([0.5, 0.0, 0.3])
      world.SetWorldTransform(new_boxes[0], box_transform)
      box_transform[:3, 3] = robot_base_pos + np.array([-0.5, 0.0, 0.3])
      world.SetWorldTransform(new_boxes[1], box_transform)

      assert not world.RobotsInCollision()

      success_count = 0

      sample_sphere_radius = 0.9
      num_samples = 500

      for _ in range(num_samples):
        # With num_samples randomly sampled poses within a sample_sphere_radius
        # sphere of the robot base, how many of them can we solve IK for?
        target_pos_offset = None
        while (target_pos_offset is None or
               np.linalg.norm(target_pos_offset) > sample_sphere_radius):
          target_pos_offset = np.random.uniform(
              low=-sample_sphere_radius, high=sample_sphere_radius, size=(3,))
        target_frame = np.eye(4)
        target_frame[:3, :3] = geometry.sample_rotation()
        target_frame[:3, 3] = target_pos_offset + robot_base_pos
        iks = world.ComputeIk(robot_name, target_frame, num_solutions=1)
        assert not world.RobotsInCollision()
        if iks:
          success_count += 1

      # Currently around 0.4 minimum. Adding some margin to avoid flakiness.
      self.assertGreater(success_count / num_samples, 0.25)


if __name__ == '__main__':
  absltest.main()
