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
from agents import actor
from agents import networks
from environments import base_env
from environments import planning_env


_EPSILON = 1e-6


class ActorsTest(absltest.TestCase):

  def _make_env_config(self):
    base_config = base_env.BaseEnvConfig(
        world_file_path='mujoco_world/4_pandas_world.xml',
        episode_timeout_s=5.0,
        timestep_s=0.5,
    )
    return planning_env.PlanningEnvConfig(
        base_config=base_config,
        num_obstacles_to_generate=10,
        num_targets_to_generate=5,
    )

  def test_acting_episode(self):
    model_config = networks.ModelConfig()
    network = networks.RoboBalletPolicyNet(config=model_config)

    env_config = self._make_env_config()

    network_variables = networks.init_network_variables(
        planning_env_config=env_config, network=network, seed=42
    )

    rng_key = jax.random.PRNGKey(42)

    trajectory, stats = actor.run_episode(
        actor_config=actor.ActorConfig(),
        feature_config=model_config.feature_config,
        planning_env_config=self._make_env_config(),
        network_apply_fn=network.apply,
        network_variables_fetch_fn=lambda: network_variables,
        rng_key=rng_key,
    )

    # Some sanity checks on the actions.
    for i, action in enumerate(trajectory.actions):
      # Actions shouldn't be identical.
      if i != 0:
        with self.assertRaises(AssertionError):
          np.testing.assert_array_almost_equal(action, trajectory.actions[0])
      # Actions should all be between min/max (for some reason np.testing only
      # has assert_array_less, not assert_array_greater). We know the joint
      # velocity limits are symmetric around 0, so we flip actions to test.
      np.testing.assert_array_less(
          action, env_config.base_config.max_joint_velocity + _EPSILON)
      np.testing.assert_array_less(
          -action, env_config.base_config.max_joint_velocity + _EPSILON)

    self.assertLen(trajectory.actions, 10)
    self.assertLen(trajectory.start_state.targets, 5)
    self.assertLen(trajectory.start_state.generated_obstacles, 10)
    self.assertLen(trajectory.observations, 11)
    self.assertLen(trajectory.rewards, 10)
    self.assertEqual(stats.episode_length, 11)

  def test_acting_random_actions(self):
    model_config = networks.ModelConfig()
    network = networks.RoboBalletPolicyNet(config=model_config)

    env_config = self._make_env_config()

    network_variables = networks.init_network_variables(
        planning_env_config=env_config, network=network, seed=42
    )

    rng_key = jax.random.PRNGKey(42)

    trajectory, stats = actor.run_episode(
        actor_config=actor.ActorConfig(),
        feature_config=model_config.feature_config,
        planning_env_config=self._make_env_config(),
        network_apply_fn=network.apply,
        network_variables_fetch_fn=lambda: network_variables,
        rng_key=rng_key,
        use_random_actions=True,
    )

    # Some sanity checks on the actions.
    for i, action in enumerate(trajectory.actions):
      # Actions shouldn't be identical.
      if i != 0:
        with self.assertRaises(AssertionError):
          np.testing.assert_array_almost_equal(action, trajectory.actions[0])
      # Actions should all be between min/max (for some reason np.testing only
      # has assert_array_less, not assert_array_greater). We know the joint
      # velocity limits are symmetric around 0, so we flip actions to test.
      np.testing.assert_array_less(
          action, env_config.base_config.max_joint_velocity + _EPSILON)
      np.testing.assert_array_less(
          -action, env_config.base_config.max_joint_velocity + _EPSILON)

    self.assertLen(trajectory.actions, 10)
    self.assertLen(trajectory.start_state.targets, 5)
    self.assertLen(trajectory.start_state.generated_obstacles, 10)
    self.assertLen(trajectory.observations, 11)
    self.assertLen(trajectory.rewards, 10)
    self.assertEqual(stats.episode_length, 11)

  def test_exp_decay(self):
    self.assertAlmostEqual(
        actor._exp_decay(initial_value=0.7, decay_rate=0.5, step=0),
        0.7)
    self.assertAlmostEqual(
        actor._exp_decay(initial_value=0.7, decay_rate=0.5, step=100000),
        0.7 * 0.5)
    self.assertAlmostEqual(
        actor._exp_decay(initial_value=0.7, decay_rate=0.5, step=200000),
        0.7 * 0.5 * 0.5)
    self.assertAlmostEqual(
        actor._exp_decay(initial_value=0.7, decay_rate=0.5, step=250000),
        0.7 * 0.5 ** 2.5)

  def test_hindsight(self):
    model_config = networks.ModelConfig()
    network = networks.RoboBalletPolicyNet(config=model_config)

    env_config = self._make_env_config()

    network_variables = networks.init_network_variables(
        planning_env_config=env_config, network=network, seed=42
    )

    rng_key = jax.random.PRNGKey(42)

    actual_trajectory, actual_stats = actor.run_episode(
        actor_config=actor.ActorConfig(),
        planning_env_config=self._make_env_config(),
        feature_config=model_config.feature_config,
        network_apply_fn=network.apply,
        network_variables_fetch_fn=lambda: network_variables,
        rng_key=rng_key,
        use_random_actions=True,
    )

    num_targets_done = np.sum(
        actual_trajectory.observations[-1]['targets']['done'])
    self.assertEqual(num_targets_done, 0)

    maybe_hindsight_trajectory_and_stats = actor.generate_hindsight_trajectory(
        actual_trajectory=actual_trajectory,
        actual_stats=actual_stats,
        planning_env_config=env_config,
        num_targets=3
    )

    self.assertIsNotNone(maybe_hindsight_trajectory_and_stats)
    hindsight_trajectory, hindsight_stats = maybe_hindsight_trajectory_and_stats

    for hindsight_action, actual_action in zip(hindsight_trajectory.actions,
                                               actual_trajectory.actions):
      np.testing.assert_array_almost_equal(hindsight_action, actual_action)

    self.assertLen(hindsight_trajectory.actions, 10)
    self.assertLen(hindsight_trajectory.start_state.targets, 5)
    self.assertLen(hindsight_trajectory.start_state.generated_obstacles, 10)
    self.assertLen(hindsight_trajectory.observations, 11)
    self.assertLen(hindsight_trajectory.rewards, 10)
    self.assertEqual(hindsight_stats.episode_length, 11)
    self.assertEqual(hindsight_stats.is_hindsight_replay, 1)

    # We cannot use hindsight_stats.num_targets_done because that records the
    # number of targets done in the actual trajectory.
    num_targets_done = np.sum(
        hindsight_trajectory.observations[-1]['targets']['done'])

    # With dwelling, hindsight trajectories are slightly broken because we may
    # put fake targets where dwelling arms are (and they can't solve it at the
    # same time). This is unlikely to be a problem in practice. We just have
    # fewer hindsight targets in some cases.
    self.assertIn(num_targets_done, [1, 2, 3])


if __name__ == '__main__':
  absltest.main()
