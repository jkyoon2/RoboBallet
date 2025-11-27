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

"""Tests for network definitions."""

from absl.testing import absltest
import jax
from jax import numpy as jnp
import jraph
import numpy as np
from agents import compute_features
from agents import networks
from environments import base_env
from environments import planning_env


class NetworksTest(absltest.TestCase):

  def _make_config(self):
    base_config = base_env.BaseEnvConfig(
        world_file_path='mujoco_world/4_pandas_world.xml',
        episode_timeout_s=5.0,
        timestep_s=0.5,
    )
    return planning_env.PlanningEnvConfig(
        base_config=base_config,
        num_obstacles_to_generate=10,
        num_targets_to_generate=2,
    )

  def _make_env(self, config):
    return planning_env.PlanningEnv(config=config)

  def _assert_arrays_not_almost_equal(self, a, b):
    with self.assertRaises(AssertionError):
      np.testing.assert_array_almost_equal(a, b, decimal=3)

  def _assert_graphs_almost_equal(self, a: jraph.GraphsTuple,
                                  b: jraph.GraphsTuple):
    for a_tensor, b_tensor in zip(a, b):
      np.testing.assert_array_almost_equal(a_tensor, b_tensor, decimal=3)

  def _randomise_graph_data(self, g: jraph.GraphsTuple) -> jraph.GraphsTuple:
    assert isinstance(g.nodes, jnp.ndarray)
    assert isinstance(g.edges, jnp.ndarray)
    assert isinstance(g.globals, jnp.ndarray)
    assert isinstance(g.senders, jnp.ndarray)
    assert isinstance(g.receivers, jnp.ndarray)
    return jraph.GraphsTuple(
        nodes=jnp.array(np.random.random_sample(size=g.nodes.shape)),
        edges=jnp.array(np.random.random_sample(size=g.edges.shape)),
        globals=jnp.array(np.random.random_sample(size=g.globals.shape)),
        senders=g.senders,
        receivers=g.receivers,
        n_node=g.n_node,
        n_edge=g.n_edge,
    )

  def test_robot_node_indices(self):
    indices = networks.get_robot_node_indices(
        num_nodes=20, num_graphs=4, num_robots=2)
    self.assertSequenceEqual(indices, [0, 1, 5, 6, 10, 11, 15, 16])
    with self.assertRaises(AssertionError):
      networks.get_robot_node_indices(
          num_nodes=20, num_graphs=6, num_robots=2)

  def test_single_network_inference(self):
    model_config = networks.ModelConfig()

    env = self._make_env(self._make_config())
    obs = env.reset().observation
    obs_spec = env.observation_spec()

    graph_features = compute_features.make_graph_features(
        obs, obs_spec, model_config.feature_config)
    network = networks.RoboBalletPolicyNet(config=model_config)
    variables = network.init(jax.random.key(42), graph_features)
    outputs = network.apply(variables, graph_features)
    self.assertSequenceEqual(outputs.shape, (1, 4, 7))

  def test_batch_network_inference(self):
    model_config = networks.ModelConfig(use_layernorm=False)

    env = self._make_env(self._make_config())
    obs = env.reset().observation
    obs_spec = env.observation_spec()

    graph_features = compute_features.make_graph_features(
        obs, obs_spec, model_config.feature_config)
    network = networks.RoboBalletPolicyNet(config=model_config)
    variables = network.init(jax.random.key(42), graph_features)

    # These asserts silence pytype warnings (since GraphsTuple components may be
    # None according to GraphsTuple type specs).
    assert isinstance(graph_features.nodes, jnp.ndarray)
    assert isinstance(graph_features.edges, jnp.ndarray)
    assert isinstance(graph_features.globals, jnp.ndarray)
    assert isinstance(graph_features.senders, jnp.ndarray)
    assert isinstance(graph_features.receivers, jnp.ndarray)

    # Make some graphs with the same shapes but random values.
    test_graph_inputs = []
    for _ in range(5):
      test_graph_inputs.append(
          jraph.GraphsTuple(
              nodes=jnp.array(
                  np.random.random_sample(size=graph_features.nodes.shape)
              ),
              edges=jnp.array(
                  np.random.random_sample(size=graph_features.edges.shape)
              ),
              globals=jnp.array(
                  np.random.random_sample(size=graph_features.globals.shape)
              ),
              senders=graph_features.senders,
              receivers=graph_features.receivers,
              n_node=graph_features.n_node,
              n_edge=graph_features.n_edge,
          )
      )

    # Sanity check - inputs shouldn't be the same.
    self._assert_arrays_not_almost_equal(
        test_graph_inputs[0].nodes, test_graph_inputs[1].nodes
    )
    self._assert_arrays_not_almost_equal(
        test_graph_inputs[1].nodes, test_graph_inputs[2].nodes
    )
    self._assert_arrays_not_almost_equal(
        test_graph_inputs[0].edges, test_graph_inputs[1].edges
    )
    self._assert_arrays_not_almost_equal(
        test_graph_inputs[1].edges, test_graph_inputs[2].edges
    )
    self._assert_arrays_not_almost_equal(
        test_graph_inputs[0].globals, test_graph_inputs[1].globals
    )
    self._assert_arrays_not_almost_equal(
        test_graph_inputs[1].globals, test_graph_inputs[2].globals
    )

    # Forwarding separately should generate the same outputs as forwarding them
    # in batch.
    separate_outputs = [
        network.apply(variables, input_graph)
        for input_graph in test_graph_inputs
    ]
    batched_output = network.apply(variables, jraph.batch(test_graph_inputs))
    self.assertSequenceEqual(batched_output.shape, (5, 4, 7))

    for i, output in enumerate(separate_outputs):
      np.testing.assert_array_almost_equal(output[0], batched_output[i])
      # Another sanity check - outputs shouldn't be the same since inputs were
      # all different.
      if i != 0:
        with self.assertRaises(AssertionError):
          np.testing.assert_array_almost_equal(output[0], batched_output[0])

  def test_single_td3_networks_inference(self):
    model_config = networks.ModelConfig()

    env = self._make_env(self._make_config())
    obs = env.reset().observation
    obs_spec = env.observation_spec()

    graph_features = compute_features.make_graph_features(
        obs, obs_spec, model_config.feature_config)
    policy_net = networks.RoboBalletPolicyNet(config=model_config)
    policy_net_variables = policy_net.init(jax.random.key(42), graph_features)
    action = policy_net.apply(policy_net_variables, graph_features)
    self.assertSequenceEqual(action.shape, (1, 4, 7))

    critic_net = networks.RoboBalletCriticNet(config=model_config)
    actions = jnp.zeros((1, 4, 7), dtype=jnp.float32)
    critic_net_variables = critic_net.init(jax.random.key(42), graph_features,
                                           actions)
    q_values = critic_net.apply(
        critic_net_variables, graph_features, actions).q_values
    self.assertSequenceEqual(q_values.shape, (1,))

  def test_network_reconstruction(self):
    env_config = self._make_config()
    env = self._make_env(env_config)
    obs = env.reset().observation
    obs_spec = env.observation_spec()

    graph = compute_features.make_graph_features(
        obs, obs_spec, compute_features.FeatureConfig())
    topology = networks.get_graph_topology(
        planning_env_config=env_config,
        feature_config=compute_features.FeatureConfig())
    reconstructed = networks.reconstruct_graph(
        nodes=np.array(graph.nodes), edges=np.array(graph.edges),
        globals_=np.array(graph.globals),
        topology=topology)
    self._assert_graphs_almost_equal(graph, reconstructed)

  def test_batch_network_reconstruction(self):
    env_config = self._make_config()
    env = self._make_env(env_config)
    obs = env.reset().observation
    obs_spec = env.observation_spec()

    graph = compute_features.make_graph_features(
        obs, obs_spec, compute_features.FeatureConfig())
    single_topology = networks.get_graph_topology(
        planning_env_config=env_config,
        feature_config=compute_features.FeatureConfig()
    )
    batched_topology = networks.get_graph_topology(
        planning_env_config=env_config, batch_size=4,
        feature_config=compute_features.FeatureConfig()
    )

    test_graphs = [self._randomise_graph_data(graph) for _ in range(4)]

    # Approach 1: Reconstruct each graph individually, and use jraph.batch()
    #             to make a batched GraphsTuple.
    individually_reconstructed = [
        networks.reconstruct_graph(
            nodes=np.array(g.nodes),
            edges=np.array(g.edges),
            globals_=np.array(g.globals),
            topology=single_topology,
        )
        for g in test_graphs
    ]

    batched_graph_from_individually_reconstructed = jraph.batch(
        individually_reconstructed
    )

    # Approach 2: Stack all the nodes, edges, globals, and reconstruct
    #             using the batched topology. This is what we do in learner for
    #             speed.
    batched_graph_from_concatenated_data = (
        networks.reconstruct_graph_from_batched_data(
            batched_nodes=np.stack([g.nodes for g in test_graphs]),
            batched_edges=np.stack([g.edges for g in test_graphs]),
            batched_globals=np.stack([g.globals for g in test_graphs]),
            batched_topology=batched_topology,
        )
    )

    # The results from the two approaches should match.
    self._assert_graphs_almost_equal(
        batched_graph_from_individually_reconstructed,
        batched_graph_from_concatenated_data,
    )


if __name__ == '__main__':
  absltest.main()
