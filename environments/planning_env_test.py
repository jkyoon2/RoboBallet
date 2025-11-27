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
from absl.testing import parameterized
from dm_env import specs
from dm_env import test_utils
import numpy as np

from environments import base_env
from environments import geometry
from environments import planning_env
from environments import target


class MujocoPlanningEnvTest(test_utils.EnvironmentTestMixin, absltest.TestCase):

  def _make_config(self, timestep=0.5, dwell_steps=0):
    base_config = base_env.BaseEnvConfig(
        world_file_path='mujoco_world/4_pandas_world.xml',
        episode_timeout_s=10 * timestep,
        timestep_s=timestep,
    )
    return planning_env.PlanningEnvConfig(
        base_config=base_config,
        randomize_robot_placement=False,
        num_obstacles_to_generate=10,
        num_targets_to_generate=2,
        target_scoring_config=target.TargetScoringConfig(
            max_shaped_score=0.5,
            distance_score_weight=0.4,
        ),
        dwell_time_s=dwell_steps * timestep,
        return_to_start_reward_weight=0.0,
        return_to_start_joint_config_diff_threshold=10.0 * np.pi,
        obstacle_spans_range=planning_env.Range(
            mins=np.array([0.02, 0.02, 0.05]), maxs=np.array([0.1, 0.1, 0.1])
        ),
        obstacle_position_range=planning_env.Range(
            mins=np.array([-0.1, -0.1, 0.0]), maxs=np.array([0.1, 0.1, 0.1])
        ),
    )

  def _make_env(self, config):
    return planning_env.PlanningEnv(config=config)

  def make_object_under_test(self):
    return self._make_env(config=self._make_config())


class PlanningEnvTest(parameterized.TestCase):

  def _make_config(self, timestep=0.5, dwell_steps=0):
    world_file_path = 'mujoco_world/4_pandas_world.xml'
    base_config = base_env.BaseEnvConfig(
        world_file_path=world_file_path,
        episode_timeout_s=10 * timestep,
        timestep_s=timestep,
    )
    return planning_env.PlanningEnvConfig(
        base_config=base_config,
        randomize_robot_placement=False,
        num_obstacles_to_generate=10,
        num_targets_to_generate=2,
        target_scoring_config=target.TargetScoringConfig(
            max_shaped_score=0.5,
            distance_score_weight=0.4,
        ),
        dwell_time_s=dwell_steps * timestep,
        return_to_start_reward_weight=0.0,
        return_to_start_joint_config_diff_threshold=10.0 * np.pi,
        obstacle_spans_range=planning_env.Range(
            mins=np.array([0.02, 0.02, 0.05]), maxs=np.array([0.1, 0.1, 0.1])
        ),
        obstacle_position_range=planning_env.Range(
            mins=np.array([-0.1, -0.1, 0.0]), maxs=np.array([0.1, 0.1, 0.1])
        ),
    )

  def _make_env(self, config):
    return planning_env.PlanningEnv(config=config)

  def _get_all_target_infos(self, env):
    (target_infos, _) = zip(*env._targets)
    return target_infos

  def _get_all_obstacle_infos(self, env):
    return list(zip(*env._generated_obstacles))[1]

  def test_observation_spec_obstacles(self):
    config = self._make_config()
    env = self._make_env(config=config)

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
            'dwelling': specs.BoundedArray(
                dtype=np.float32,
                shape=(4,),
                minimum=0.0,
                maximum=1.0,
            ),
            'robot_index': specs.BoundedArray(
                dtype=np.float32,
                shape=(4, 4),
                minimum=0,
                maximum=1,
            ),
        },
        'obstacles': {
            'spans': specs.BoundedArray(
                dtype=np.float32,
                shape=(10, 3),
                minimum=-base_env.WORLD_SPAN_M / 2.0,
                maximum=base_env.WORLD_SPAN_M / 2.0,
            ),
            'poses': specs.BoundedArray(
                dtype=np.float32,
                shape=(10, 4, 4),
                minimum=base_env.POSE_BOUNDS_MIN,
                maximum=base_env.POSE_BOUNDS_MAX,
            ),
        },
        'targets': {
            'poses': specs.BoundedArray(
                dtype=np.float32,
                shape=(2, 4, 4),
                minimum=base_env.POSE_BOUNDS_MIN,
                maximum=base_env.POSE_BOUNDS_MAX,
            ),
            'done': specs.BoundedArray(
                dtype=bool, shape=(2,), minimum=False, maximum=True
            ),
        },
        'current_score': specs.BoundedArray(
            shape=(), dtype=np.float32, minimum=0.0, maximum=1.0
        ),
    }
    self.assertEqual(env.observation_spec(), expected_observation_spec)

  def test_stepping_rewards(self):
    config = self._make_config()
    config = config.replace(collision_penalty_scaling=0.0)
    env = self._make_env(config=config)
    env.reset()
    starting_score = env._score_and_update_all_targets()
    self.assertGreater(starting_score, 0.0)
    step0 = env.step(np.zeros((4, 7)))
    # The first reward should be equal to the score, because reset step doesn't
    # have a reward.
    self.assertAlmostEqual(step0.reward, starting_score)
    step1 = env.step(np.ones((4, 7)))
    new_score = env._score_and_update_all_targets()
    self.assertNotAlmostEqual(new_score, starting_score)
    # The second reward on should be score differences.
    self.assertAlmostEqual(step1.reward, new_score - starting_score)

    rewards = []
    while not env.terminal():
      step = env.step(np.random.rand(4, 7) - 0.5)
      self.assertGreaterEqual(step.reward, -1.0)
      self.assertLessEqual(step.reward, 1.0)
      rewards.append(step.reward)

    # Rewards should add up to the final score if there are no collisions.
    reward_sum = step0.reward + step1.reward + sum(rewards)
    final_score = env._score_and_update_all_targets()
    self.assertGreaterEqual(final_score, 0.0)
    self.assertLessEqual(final_score, 1.0)
    self.assertAlmostEqual(final_score, reward_sum)

  def test_collision_penalty(self):
    config = self._make_config()
    config = config.replace(collision_penalty_scaling=42.0)
    env = self._make_env(config=config)
    env.reset()
    action = np.zeros(shape=(4, 7), dtype=np.float32)
    action[0, :] = np.array([0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0])
    # Crash panda1 into the table and keep going. We should stop at
    # qpos[1] == 0.4146
    num_robots_in_collision = []
    for _ in range(10):
      env.step(action)
      num_robots_in_collision.append(len(env._robots_in_collision_last_step))
    self.assertSequenceEqual(
        num_robots_in_collision, [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    )
    self.assertAlmostEqual(
        env.total_collision_penalty(), 42.0 * (1.0 / 4.0) * (6.0 / 10.0)
    )
    self.assertEqual(env.total_collision_robot_steps(), 6)

  def test_acceleration_penalty(self):
    config = self._make_config()
    config = config.replace(
        acceleration_cost=3e-4,
        base_config=config.base_config.replace(timestep_s=0.1),
        target_scoring_config=target.TargetScoringConfig(max_shaped_score=0.0),
    )
    env = self._make_env(config=config)
    env.reset()
    action = np.zeros(shape=(4, 7), dtype=np.float32)
    # The velocity here should have been clipped to 0.8 rad/s.
    action[0, :] = np.array([0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    env.step(action)

    # Now 1.2
    action[0, :] = np.array([0.0, 1.2, 1.2, 0.0, 0.0, 0.0, 0.0])
    new_timestep = env.step(action)

    # Now the total qvel delta is 0.4 rad/s, and acceleration is 4.0 rad/s^2.
    # avg_qacc2 should be (4.0 ** 2) * 2 / 28 = 1.142857.
    # Acceleration cost scaled by targets should be 3e-4/2 = 1.5e-4
    # Acceleration penalty should be 1.5e-4 * 1.142857 * 0.1 = 1.714285e-5

    self.assertAlmostEqual(new_timestep.reward, -1.714285e-5, places=8)

    # Acceleration cost:
    # Step 1: 8.0 * 2 / 28 * 0.1 = 0.05714286
    # Step 2: 4.0 * 2 / 28 * 0.1 = 0.02857143
    # Sum: 0.0857143
    self.assertAlmostEqual(env.total_acceleration(), 0.0857143, places=5)

  def test_start_state_serialization(self):
    config = self._make_config()
    env = self._make_env(config)
    env.reset()
    start_state = env.get_start_state()
    new_env_without_start_state = self._make_env(config=config)
    new_env_without_start_state.reset()
    with self.assertRaises(AssertionError):
      self.assertSequenceEqual(
          self._get_all_target_infos(env),
          self._get_all_target_infos(new_env_without_start_state),
      )
    with self.assertRaises(AssertionError):
      self.assertSequenceEqual(
          self._get_all_obstacle_infos(env),
          self._get_all_obstacle_infos(new_env_without_start_state),
      )
    new_env_with_start_state = planning_env.PlanningEnv(
        config=config, start_state=start_state
    )
    new_env_with_start_state.reset()
    self.assertSequenceEqual(
        self._get_all_target_infos(env),
        self._get_all_target_infos(new_env_with_start_state),
    )
    self.assertSequenceEqual(
        self._get_all_obstacle_infos(env),
        self._get_all_obstacle_infos(new_env_with_start_state),
    )

  def test_sampling_objects(self):
    config = self._make_config()
    env = self._make_env(config)
    env.reset()
    eligible_objs = env._eligible_objects_for_sampled_targets()
    self.assertLen(eligible_objs, config.num_obstacles_to_generate)
    for obj in eligible_objs:
      self.assertStartsWith(obj, 'box_')

  def test_box_augmentation(self):
    box_info = planning_env.BoxInfo(
        spans=np.array([2.0, 4.0, 6.0]),
        centroid=np.array([4.0, 5.0, 6.0]),
        rotation=geometry.sample_rotation(),
    )
    augmented_box_infos = [box_info.get_augmentation(i) for i in range(4)]
    assert augmented_box_infos[0] == box_info
    np.testing.assert_allclose(
        augmented_box_infos[1].spans, np.array([2.0, 6.0, 4.0])
    )
    np.testing.assert_allclose(
        augmented_box_infos[2].spans, np.array([6.0, 4.0, 2.0])
    )
    np.testing.assert_allclose(
        augmented_box_infos[3].spans, np.array([4.0, 2.0, 6.0])
    )

    # Now we add all these boxes to the world, and check that they all end up
    # with the same transformed vertices.
    config = self._make_config()
    env = self._make_env(config=config)
    env.reset()

    def add_box(box_info_to_add):
      box_id = env._world.AddBox(box_info_to_add.spans)
      world_t_box = np.eye(4)
      world_t_box[:3, :3] = box_info_to_add.rotation
      world_t_box[:3, 3] = box_info_to_add.centroid
      env._world.SetWorldTransform(box_id, world_t_box)
      return box_id

    all_objs = [add_box(bi) for bi in augmented_box_infos]

    def get_vertices(world, obj_id: str):
      mesh = world.GetMeshes(obj_id)[0]
      return mesh.vertices[np.lexsort(np.transpose(mesh.vertices))]

    for aug_sel in range(1, 4):
      np.testing.assert_allclose(
          get_vertices(env._world, all_objs[0]),
          get_vertices(env._world, all_objs[aug_sel]),
          atol=1e-5,
      )

  def _replace_target_info(self, env, target_num, new_target_info):
    _, old_target_state = env._targets[target_num]
    env._targets[target_num] = (new_target_info, old_target_state)

  def test_target_scoring(self):
    config = self._make_config()
    env = self._make_env(config)
    env.reset()
    robots = env._robots
    robot_tcps = {
        robot: env._world.GetRobotTcpTransform(robot) for robot in robots
    }
    # TCPs should not be the same.
    for robot_i in robots:
      for robot_j in robots:
        if robot_i != robot_j:
          with self.assertRaises(AssertionError):
            np.testing.assert_allclose(robot_tcps[robot_i], robot_tcps[robot_j])

    # Test scoring and target state. Instead of moving arms, we move targets
    # here for simplicity.
    # Robot tooltips are at:
    # panda1: [ 0.31328, -0.6285, 0.51928]
    # panda2: [-0.31328, -0.6285, 0.51928]
    # panda3: [-0.31328,  0.6285, 0.51928]
    # panda4: [ 0.31328,  0.6285, 0.51928]

    # Targets are now each between 2 robots, at distance=0.31328
    self._replace_target_info(
        env,
        target_num=0,
        new_target_info=target.TargetInfo(
            position=np.array([0.0, 0.6285, 0.51928])
        ),
    )
    self._replace_target_info(
        env,
        target_num=1,
        new_target_info=target.TargetInfo(
            position=np.array([0.0, -0.6285, 0.51928])
        ),
    )

    # Target score should be max_shaped_score * coeff / (coeff + distance) when
    # the target is outside the solution radius of every arm, and 1.0 when
    # inside (or if the target has already been solved before).
    # Currently max_shaped_score = coeff = 0.5.
    # The total environment score is the average of the target scores.
    # Eg. 0.5 * 0.5 / (0.5 + 0.3133) = 0.307397
    self.assertAlmostEqual(
        env._score_and_update_all_targets(),
        (0.307397 + 0.307397) / 2,
        delta=1e-3,
    )

    # Move target 1 closer to one arm (by 0.1).
    self._replace_target_info(
        env,
        target_num=1,
        new_target_info=target.TargetInfo(
            position=np.array([0.1, -0.6285, 0.51928])
        ),
    )

    # Now it should have a higher score.
    self.assertAlmostEqual(
        env._score_and_update_all_targets(),
        (0.307397 + 0.35049) / 2,
        delta=1e-3,
    )

    # Move the target again to within the solution radius.
    self._replace_target_info(
        env,
        target_num=1,
        new_target_info=target.TargetInfo(
            position=np.array([0.315, -0.6285, 0.51928])
        ),
    )

    self.assertAlmostEqual(
        env._score_and_update_all_targets(), (0.307397 + 1.0) / 2, delta=1e-3
    )
    self.assertTrue(env._targets[1][1].done)

    # When we move it away again, it should stay done.
    self._replace_target_info(
        env,
        target_num=1,
        new_target_info=target.TargetInfo(
            position=np.array([0.0, -0.6285, 0.51928])
        ),
    )

    self.assertAlmostEqual(
        env._score_and_update_all_targets(), (0.307397 + 1.0) / 2, delta=1e-3
    )
    self.assertTrue(env._targets[1][1].done)

  def test_dwelling(self):
    config = self._make_config(dwell_steps=2)
    env = self._make_env(config=config)
    env.reset()
    robot_tcps = {
        robot: env._world.GetRobotTcpTransform(robot) for robot in env._robots
    }
    timestep = env.step(np.zeros((4, 7), dtype=np.float32))
    np.testing.assert_allclose(
        timestep.observation['robots']['dwelling'],
        np.array([0.0, 0.0, 0.0, 0.0]),
    )

    # Move a target to the tcp of robot 1
    self._replace_target_info(
        env,
        target_num=0,
        new_target_info=target.TargetInfo(
            position=robot_tcps[env._robots[1]][:3, 3]
        ),
    )

    timestep = env.step(np.zeros((4, 7), dtype=np.float32))
    robot_tcps_before = {
        robot: env._world.GetRobotTcpTransform(robot) for robot in env._robots
    }
    self.assertTrue(env._targets[0][1].done)

    # Now robot 1 should be dwelling.
    np.testing.assert_allclose(
        timestep.observation['robots']['dwelling'],
        np.array([0.0, 1.0, 0.0, 0.0]),
    )

    # If we tell all robots to move, robot 1 shouldn't move.
    timestep = env.step(np.ones((4, 7), dtype=np.float32) * 0.01)
    robot_tcps_after = {
        robot: env._world.GetRobotTcpTransform(robot) for robot in env._robots
    }
    np.testing.assert_allclose(
        timestep.observation['robots']['dwelling'],
        np.array([0.0, 0.5, 0.0, 0.0]),
    )
    for robot in env._robots:
      if robot != env._robots[1]:
        with self.assertRaises(AssertionError):
          np.testing.assert_allclose(
              robot_tcps_before[robot], robot_tcps_after[robot]
          )
      else:
        np.testing.assert_allclose(
            robot_tcps_before[robot], robot_tcps_after[robot]
        )

    # One more time.
    robot_tcps_before = robot_tcps_after
    timestep = env.step(np.ones((4, 7), dtype=np.float32) * 0.01)
    robot_tcps_after = {
        robot: env._world.GetRobotTcpTransform(robot) for robot in env._robots
    }
    np.testing.assert_allclose(
        timestep.observation['robots']['dwelling'],
        np.array([0.0, 0.0, 0.0, 0.0]),
    )
    for robot in env._robots:
      if robot != env._robots[1]:
        with self.assertRaises(AssertionError):
          np.testing.assert_allclose(
              robot_tcps_before[robot], robot_tcps_after[robot]
          )
      else:
        np.testing.assert_allclose(
            robot_tcps_before[robot], robot_tcps_after[robot]
        )

    # Now it should move again.
    robot_tcps_before = robot_tcps_after
    timestep = env.step(np.ones((4, 7), dtype=np.float32) * 0.01)
    robot_tcps_after = {
        robot: env._world.GetRobotTcpTransform(robot) for robot in env._robots
    }
    np.testing.assert_allclose(
        timestep.observation['robots']['dwelling'],
        np.array([0.0, 0.0, 0.0, 0.0]),
    )
    for robot in env._robots:
      with self.assertRaises(AssertionError):
        np.testing.assert_allclose(
            robot_tcps_before[robot], robot_tcps_after[robot]
        )

  def test_random_robot_placements(self):
    config = self._make_config()
    config = config.replace(randomize_robot_placement=True)
    env = self._make_env(config=config)
    env.reset()
    panda1_pose = env._world.GetWorldTransform('panda1')
    panda1_position = panda1_pose[:3, 3]
    self.assertAlmostEqual(panda1_position[0], 0.5515)
    self.assertNotAlmostEqual(panda1_position[1], -0.6285)
    self.assertLessEqual(panda1_position[1], 0.8)
    self.assertGreaterEqual(panda1_position[1], -0.8)
    self.assertAlmostEqual(panda1_position[2], 0.009)

  def test_random_envs_with_random_robot_placements(self):
    config = self._make_config()
    config = config.replace(randomize_robot_placement=True)
    config = config.replace(num_obstacles_to_generate=4)
    config = config.replace(num_targets_to_generate=2)
    for _ in range(5):
      env = self._make_env(config=config)
      env.reset()

  def test_static_obstacles(self):
    config = self._make_config()
    config = config.replace(num_obstacles_to_generate=8)
    config = config.replace(num_targets_to_generate=2)
    config = config.replace(obstacles_by_decomposition={'workpiece': 10})
    env = self._make_env(config=config)
    env.reset()
    obs_spec = env.observation_spec()
    obs = env.observation()
    self.assertEqual(obs_spec['obstacles']['spans'].shape[0], 18)
    self.assertEqual(obs['obstacles']['spans'].shape[0], 18)

    start_state = env.get_start_state()
    self.assertLen(start_state.generated_obstacles, 18)
    env = planning_env.PlanningEnv(config=config, start_state=start_state)
    env.reset()
    obs_spec = env.observation_spec()
    obs = env.observation()
    self.assertEqual(obs_spec['obstacles']['spans'].shape[0], 18)
    self.assertEqual(obs['obstacles']['spans'].shape[0], 18)

  def test_return_to_start_scoring(self):
    config = self._make_config()
    config = config.replace(
        return_to_start_reward_weight=0.4,
        return_to_start_joint_config_diff_threshold=0.1,
        target_scoring_config=target.TargetScoringConfig(max_shaped_score=0.0),
    )
    env = self._make_env(config=config)
    env.reset()

    start_joint_config = env._world.GetJointValues('panda1')

    # We should have 0 score to start with, even though the arms are at the
    # start config, because we haven't solved all targets yet.
    self.assertAlmostEqual(env._score_and_update_all_targets(), 0.0)

    # Solve one target.
    # Test scoring and target state. Instead of moving arms, we move targets
    # here for simplicity.
    # Robot tooltips are at:
    # panda1: [ 0.31328, -0.6285, 0.51928]
    # panda2: [-0.31328, -0.6285, 0.51928]
    # panda3: [-0.31328,  0.6285, 0.51928]
    # panda4: [ 0.31328,  0.6285, 0.51928]
    self._replace_target_info(
        env,
        target_num=0,
        new_target_info=target.TargetInfo(
            position=np.array([0.31328, -0.6285, 0.51928])
        ),
    )

    # Now we have the reward for solving the target, but still no return to
    # start reward.
    self.assertAlmostEqual(env._score_and_update_all_targets(), 0.3)

    # Now move two of the arms somewhere else before we solve the second (last)
    # target.
    env._world.SetJointValues('panda3', start_joint_config + 0.15)
    env._world.SetJointValues('panda4', start_joint_config - 0.15)

    # This shouldn't change the score.
    self.assertAlmostEqual(env._score_and_update_all_targets(), 0.3)

    # Now we solve the last target.
    self._replace_target_info(
        env,
        target_num=1,
        new_target_info=target.TargetInfo(
            position=np.array([-0.31328, -0.6285, 0.51928])
        ),
    )

    self.assertFalse(env.terminal())

    # We should have the return to start reward now.
    shaped_rts_score = 1.0 - 3.0 * 0.15**2
    rts_score = 1.0 + 1.0 + shaped_rts_score + shaped_rts_score
    rts_score /= 4
    self.assertAlmostEqual(env._return_to_start_score(), rts_score)
    self.assertAlmostEqual(
        env._score_and_update_all_targets(), 0.6 + 0.4 * rts_score
    )

    # Return all arms to within threshold.
    env._world.SetJointValues('panda3', start_joint_config + 0.07)
    env._world.SetJointValues('panda4', start_joint_config - 0.07)

    # Now we should be getting 1.0 score, but the episode is not done yet,
    # because the hardcoded return to start controller still needs to actually
    # park the arms.
    self.assertAlmostEqual(env._score_and_update_all_targets(), 1.0)
    self.assertFalse(env.terminal())

    # Take a step. The value doesn't matter because the return to start
    # controller will overwrite it anyways.
    env.step(np.zeros((4, 7), dtype=np.float32))
    self.assertTrue(env.terminal())

  def test_setting_starting_joint_config(self):
    config = self._make_config()
    config = config.replace(
        return_to_start_reward_weight=0.4,
        return_to_start_joint_config_diff_threshold=0.1,
        target_scoring_config=target.TargetScoringConfig(max_shaped_score=0.0),
        num_targets_to_generate=1,
        starting_joint_config=[
            np.array([0.0, -0.7, 0.0, -0.5, 0.0, 1.0, 0.0]),
            np.array([0.0, -0.785, 0.0, -2.182, 0.0, 1.396, 0.0]),
            np.array([0.1, -0.1, 0.1, -0.1, 0.0, 1.0, 0.1]),
            np.array([0.1, -0.2, 0.3, -2.182, 0.0, 1.5, 0.0]),
        ],
    )
    env = self._make_env(config=config)
    reset_step = env.reset()

    reset_obs = reset_step.observation
    self.assertSequenceAlmostEqual(
        reset_obs['robots']['joint_configurations'][0, :],
        [0.0, -0.7, 0.0, -0.5, 0.0, 1.0, 0.0],
        places=3,
    )

    start_state = env.get_start_state()
    self.assertSequenceAlmostEqual(
        start_state.initial_joint_configurations[0],
        [0.0, -0.7, 0.0, -0.5, 0.0, 1.0, 0.0],
        places=3,
    )

    # We should have 0 score to start with, even though the arms are at the
    # start config, because we haven't solved all targets yet.
    self.assertAlmostEqual(env._score_and_update_all_targets(), 0.0)

    # Move the arms out of start config.
    for i, robot in enumerate(env._robots):
      env._world.SetJointValues(robot, config.starting_joint_config[i] - 0.3)

    # Solve one target.
    # Test scoring and target state. Instead of moving arms, we move targets
    # here for simplicity.
    # Robot tooltips are at:
    # panda2: [-0.52197391 -0.77108693  0.44313705]
    self._replace_target_info(
        env,
        target_num=0,
        new_target_info=target.TargetInfo(
            position=np.array([-0.52197391, -0.77108693, 0.44313705])
        ),
    )

    # Now we have the reward for solving the target, but still no return to
    # start reward.
    self.assertAlmostEqual(env._score_and_update_all_targets(), 0.892, places=3)

    # Return all arms to within threshold.
    for i, robot in enumerate(env._robots):
      env._world.SetJointValues(robot, config.starting_joint_config[i] - 0.05)

    # Now we should be getting 1.0 score, but the episode is not done yet,
    # because the hardcoded return to start controller still needs to actually
    # park the arms.
    self.assertAlmostEqual(env._score_and_update_all_targets(), 1.0)
    self.assertFalse(env.terminal())

    # Take a step. The value doesn't matter because the return to start
    # controller will overwrite it anyways.
    env.step(np.zeros((4, 7), dtype=np.float32))
    self.assertTrue(env.terminal())

  def test_sample_and_add_boxes(self):
    config = self._make_config()
    config = config.replace(num_obstacles_to_generate=2)
    env = self._make_env(config=config)
    env.reset()
    self.assertLen([env._world.GetObjects()
                    for obj in env._world.GetObjects()
                    if obj.startswith('box_')], 2)
    self.assertLen(env._generated_obstacles, 2)


if __name__ == '__main__':
  absltest.main()
