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

"""Neural network definitions."""

from collections.abc import Sequence
from typing import Any

from flax import struct
import flax.linen as nn
import jax
import jax.numpy as jnp
import jraph
import numpy as np

from agents import compute_features
from environments import planning_env


class ModelConfig(struct.PyTreeNode):
  """Network hyper-parameters.

  Attributes:
    feature_config: Config for network features.
    mlp_layer_size: Inner layer and embedding size for MLPs (excl edge update).
    embedding_num_layers: Number of layers for the embedding networks (excl edge
      embedding).
    update_num_layers: Number of layers for the update networks (excl edge
      update).
    edge_mlp_layer_size: Inner layer size for the edge embedding and
      update networks.
    edge_embedding_num_layers: Number of layers for the edgeembedding network.
    edge_update_num_layers: Number of layers for the edge update network.
    num_message_passing_rounds: Number of message passing rounds.
    scalar_prediction_layer_sizes: Scalar output head MLP layer sizes.
    policy_prediction_layer_sizes: Policy output head MLP layer sizes.
    policy_prediction_num_bins: Number of bins for the probability density
      function prediction per robot degree of freedom.
    num_robots: Number of robots.
    action_dims_per_robot: Number of degrees of freedom per robot.
    last_layer_mlp_scale: Initialisation scale for the last layer of MLP before
      tanh in actor and critic networks. This keeps more of the MLP output range
      in the part of tanh with higher gradient, and speeds up initial training.
    use_layernorm: Use layer norm.
    use_attention: Use edge attention.
    activation_type: Activation function ('relu', 'elu', 'lrelu', or 'gelu')
    graph_skip_connections: Graph-level skip connections.
    shared_weight_message_passing: Shared weights between message passing
      rounds.
    use_bf16: Use bfloat16 where sensible (not for accumulations).
    aggregate_fn: Graph update aggregation function ('sum' or 'mean')
  """

  feature_config: compute_features.FeatureConfig = (
      compute_features.FeatureConfig())

  mlp_layer_size: int = 512
  embedding_num_layers: int = 7
  update_num_layers: int = 7

  # We have a lot more edge updates than node/global updates, so have separately
  # tuned number of layers and layer size.
  # In a message passing round with 8 arms, 40 targets, and 24 obstacles, we
  # have 568 edge updates, 8 node updates, and 1 global update.
  edge_mlp_layer_size: int = 256
  edge_embedding_num_layers: int = 6
  edge_update_num_layers: int = 7

  num_message_passing_rounds: int = 1

  # Layer sizes for prediction head MLPs.
  scalar_prediction_layer_sizes: Sequence[int] = (64, 64)
  policy_prediction_layer_sizes: Sequence[int] = (64, 64)

  # Policy output is a discrete probability density function.
  policy_prediction_num_bins: int = 21
  num_robots: int = 4
  action_dims_per_robot: int = 7

  last_layer_mlp_scale: float = 0.005

  use_layernorm: bool = True

  use_attention: bool = False

  activation_type: str = 'gelu'  # 'relu', 'elu', 'lrelu', or 'gelu'

  graph_skip_connections: bool = False
  shared_weight_message_passing: bool = True

  use_bf16: bool = True
  aggregate_fn: str = 'sum'

  def update_for_cpu_inference(self) -> 'ModelConfig':
    """Update config for CPU inference."""

    # Ignore the use_bf16 setting if we are on CPU. As of this writing, bf16 on
    # CPU is almost 20x slower than f32! This may change in the future if we
    # start getting CPUs with bf16 support. AVX-512 supposedly supports it, but
    # we need support in the whole stack from Eigen (Jax CPU backend) to
    # XLA:CPU for that to work.
    return self.replace(use_bf16=False)


class MLP(nn.Module):
  """Regular MLP."""

  layer_sizes: Sequence[int]
  config: ModelConfig
  last_layer_mlp_scale: float = 1.0

  @nn.compact
  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    dtype = jnp.bfloat16 if self.config.use_bf16 else None
    x = inputs.astype(dtype=dtype)
    for i, layer_size in enumerate(self.layer_sizes):
      is_last_layer = i == len(self.layer_sizes) - 1
      if is_last_layer:
        # This is the same as the default lecun_normal initialiser, but with
        # a different scale.
        last_layer_kernel_init = nn.initializers.variance_scaling(
            scale=self.last_layer_mlp_scale, mode='fan_in',
            distribution='truncated_normal', dtype=dtype)
        x = nn.Dense(layer_size, kernel_init=last_layer_kernel_init,
                     dtype=dtype)(x)
        # No activation or layer norm on the last layer.
      else:
        x = nn.Dense(layer_size, dtype=dtype)(x)
        if self.config.use_layernorm:
          x = nn.LayerNorm(dtype=dtype)(x)
        if self.config.activation_type == 'relu':
          x = nn.relu(x)
        elif self.config.activation_type == 'elu':
          x = nn.elu(x)
        elif self.config.activation_type == 'lrelu':
          x = nn.leaky_relu(x)
        elif self.config.activation_type == 'gelu':
          x = nn.gelu(x)
    # Cast to f32 here because the output will be accumulated.
    return x.astype(jnp.float32)


class AttentionMLP(nn.Module):
  """MLP for attention logits."""

  @nn.compact
  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    # Following the GAN paper (https://arxiv.org/pdf/1710.10903.pdf), we have
    # a single layer MLP here with LeakyReLU (negative slope = 0.2).
    x = nn.Dense(1)(inputs)
    return nn.leaky_relu(x, negative_slope=0.2)


class GraphNetCore(nn.Module):
  """Graph net message passing round that assumes same shape inputs/outputs."""

  config: ModelConfig

  @nn.compact
  def __call__(self, embedding: jraph.GraphsTuple) -> jraph.GraphsTuple:
    update_edge_fn = jraph.concatenated_args(
        MLP(
            layer_sizes=[self.config.edge_mlp_layer_size]
            * self.config.edge_update_num_layers,
            config=self.config,
            name='update_edge',
        )
    )
    update_node_fn = jraph.concatenated_args(
        MLP(
            layer_sizes=[self.config.mlp_layer_size]
            * self.config.update_num_layers,
            config=self.config,
            name='update_node',
        )
    )
    update_global_fn = jraph.concatenated_args(
        MLP(
            layer_sizes=[self.config.mlp_layer_size]
            * self.config.update_num_layers,
            config=self.config,
            name='update_global',
        )
    )

    attention_logit_fn = None
    attention_reduce_fn = None
    if self.config.use_attention:
      attention_logit_fn = jraph.concatenated_args(
          AttentionMLP(name='attention_logit')
      )
      attention_reduce_fn = lambda edges, weights: edges * weights

    if self.config.aggregate_fn == 'sum':
      aggregate_fn = jraph.segment_sum
    elif self.config.aggregate_fn == 'mean':
      aggregate_fn = jraph.segment_mean
    else:
      raise ValueError(
          f'Unsupported aggregate_fn: {self.config.aggregate_fn}')

    gnn = jraph.GraphNetwork(
        update_edge_fn=update_edge_fn,
        update_node_fn=update_node_fn,
        update_global_fn=update_global_fn,
        attention_logit_fn=attention_logit_fn,
        attention_reduce_fn=attention_reduce_fn,
        aggregate_edges_for_nodes_fn=aggregate_fn,
        aggregate_nodes_for_globals_fn=aggregate_fn,
        aggregate_edges_for_globals_fn=aggregate_fn,
    )

    out_embedding = gnn(embedding)
    if self.config.graph_skip_connections:
      out_embedding = out_embedding._replace(
          nodes=embedding.nodes + out_embedding.nodes,
          edges=embedding.edges + out_embedding.edges,
          globals=embedding.globals + out_embedding.globals,
      )
    return out_embedding


class GraphNet(nn.Module):
  """Graph net using MLP update functions."""

  config: ModelConfig

  @nn.compact
  def __call__(self, graph_in: jraph.GraphsTuple) -> jraph.GraphsTuple:
    embed_edge_fn = jraph.concatenated_args(
        MLP(
            layer_sizes=[self.config.edge_mlp_layer_size]
            * self.config.edge_embedding_num_layers,
            config=self.config,
            name='embed_edge',
        )
    )
    embed_node_fn = jraph.concatenated_args(
        MLP(
            layer_sizes=[self.config.mlp_layer_size]
            * self.config.embedding_num_layers,
            config=self.config,
            name='embed_node',
        )
    )
    embed_global_fn = jraph.concatenated_args(
        MLP(
            layer_sizes=(
                [self.config.mlp_layer_size] * self.config.embedding_num_layers
            ),
            config=self.config,
            name='embed_global',
        )
    )

    embed_fn = jraph.GraphMapFeatures(
        embed_edge_fn=embed_edge_fn,
        embed_node_fn=embed_node_fn,
        embed_global_fn=embed_global_fn,
    )

    gnns = []
    if self.config.shared_weight_message_passing:
      gnns.append(GraphNetCore(config=self.config))
    else:
      for _ in range(self.config.num_message_passing_rounds):
        gnns.append(GraphNetCore(config=self.config))

    embedding = embed_fn(graph_in)
    for round_num in range(self.config.num_message_passing_rounds):
      gnn = (gnns[0]
             if self.config.shared_weight_message_passing else gnns[round_num])
      input_embedding = embedding
      output_embedding = gnn(input_embedding)
      if self.config.graph_skip_connections:
        embedding = embedding._replace(
            nodes=input_embedding.nodes + output_embedding.nodes,
            edges=input_embedding.edges + output_embedding.edges,
            globals=input_embedding.globals + output_embedding.globals,
        )
      else:
        embedding = output_embedding
    return embedding


def get_robot_node_indices(num_nodes: int, num_graphs: int,
                           num_robots: int) -> list[int]:
  """Returns node indices for robot nodes out of all nodes."""

  # Robot nodes are the first n_robot nodes of each graph.
  # We cannot use GraphsTuple.n_node here because this function needs to be
  # compilable.
  # But we know that the graphs all have the same number of nodes, so we can
  # just do a division.
  indices = []
  assert num_nodes % num_graphs == 0
  nodes_per_graph = num_nodes // num_graphs
  for i in range(num_graphs):
    node_offset = i * nodes_per_graph
    indices.extend(range(node_offset, node_offset + num_robots))
  return indices


def get_robot_nodes(g: jraph.GraphsTuple, num_robots: int) -> jnp.ndarray:
  """Returns robot nodes from the graph."""
  nodes = g.nodes
  assert nodes is not None and isinstance(nodes, jnp.ndarray)
  all_robot_indices = get_robot_node_indices(
      num_nodes=nodes.shape[0], num_graphs=g.n_node.shape[0],
      num_robots=num_robots)
  return jnp.take(
      nodes, indices=np.array(all_robot_indices), axis=0,
      unique_indices=True, indices_are_sorted=True)


def add_actions_to_robot_nodes(
    graphs_in: jraph.GraphsTuple, config: ModelConfig, actions: jnp.ndarray
) -> jraph.GraphsTuple:
  """Adds actions to the robot nodes of the graph."""
  nodes = graphs_in.nodes
  assert nodes is not None and isinstance(nodes, jnp.ndarray)
  num_graphs = graphs_in.n_node.shape[0]
  original_num_features = nodes.shape[1]
  assert actions.shape == (
      num_graphs, config.num_robots, config.action_dims_per_robot)

  # First we pad node features to add space for the actions.
  pad_widths = [(0, 0) for _ in range(len(nodes.shape))]
  pad_widths[-1] = (0, config.action_dims_per_robot)
  padded_nodes = jnp.pad(nodes, pad_width=pad_widths, mode='constant',
                         constant_values=0.0)
  # Append the actions to the end of the corresponding robot node features.
  robot_node_indices = get_robot_node_indices(
      num_nodes=nodes.shape[0],
      num_graphs=num_graphs,
      num_robots=config.num_robots,
  )
  new_nodes = padded_nodes.at[np.array(robot_node_indices),
                              original_num_features:].set(
                                  actions.reshape(-1, actions.shape[-1]))
  assert new_nodes.shape[1] == (
      original_num_features + config.action_dims_per_robot
  )

  # Replace the input graph nodes with our nodes with actions.
  return graphs_in._replace(nodes=new_nodes)


class RoboBalletPolicyNet(nn.Module):
  """Module for TD3 policy implementation."""

  config: ModelConfig

  @nn.compact
  def __call__(self, graphs_in: jraph.GraphsTuple) -> jnp.ndarray:  # pytype: disable=signature-mismatch  # jax-ndarray
    gnn_out = GraphNet(config=self.config)(graphs_in)
    num_graphs = gnn_out.n_node.shape[0]
    all_robot_nodes = get_robot_nodes(g=gnn_out,
                                      num_robots=self.config.num_robots)
    policy_layer_sizes = list(
        self.config.policy_prediction_layer_sizes) + [
            self.config.action_dims_per_robot
        ]
    policy_fn = MLP(
        layer_sizes=policy_layer_sizes,
        last_layer_mlp_scale=self.config.last_layer_mlp_scale,
        config=self.config,
        name='policy_head')
    actions = policy_fn(all_robot_nodes).reshape((
        num_graphs,
        self.config.num_robots,
        self.config.action_dims_per_robot,
    ))
    normalised_actions = jax.nn.tanh(actions)  # [-1, 1]
    return normalised_actions


@struct.dataclass
class CriticNetworkOutputs:
  """All critic network outputs.

  Attributes:
    q_values: Actual outputs.
    rewards: Reward predictions (for aux loss).
  """

  q_values: jnp.ndarray
  rewards: jnp.ndarray


class RoboBalletCriticNet(nn.Module):
  """Module for TD3 Q function critic implementation."""

  config: ModelConfig

  @nn.compact
  def __call__(self, graphs_in: jraph.GraphsTuple,  # pytype: disable=signature-mismatch  # jax-ndarray
               actions: jnp.ndarray) -> CriticNetworkOutputs:
    graphs_in = add_actions_to_robot_nodes(
        graphs_in=graphs_in, config=self.config, actions=actions)

    gnn_out = GraphNet(config=self.config)(graphs_in)

    q_values = MLP(
        layer_sizes=list(self.config.scalar_prediction_layer_sizes) + [1],
        last_layer_mlp_scale=self.config.last_layer_mlp_scale,
        config=self.config,
        name='qvalue')(gnn_out.globals)

    rewards = MLP(
        layer_sizes=list(self.config.scalar_prediction_layer_sizes) + [1],
        last_layer_mlp_scale=self.config.last_layer_mlp_scale,
        config=self.config,
        name='rewards')(gnn_out.globals)

    # Our rewards are score differences, so total future return is bounded by
    # the range of score, which is [0, 1]. Therefore ground truth q-values will
    # always be [-1, 1].
    return CriticNetworkOutputs(
        q_values=jnp.tanh(q_values[:, 0]),
        rewards=jnp.tanh(rewards[:, 0]))


class RoboBalletTwinCriticNet(nn.Module):
  """TD3 Twin Q function critic implementation."""

  config: ModelConfig

  @nn.compact
  def __call__(self, graphs_in: jraph.GraphsTuple,
               actions: jnp.ndarray) -> tuple[CriticNetworkOutputs,
                                              CriticNetworkOutputs]:
    critic_1 = RoboBalletCriticNet(config=self.config, name='critic_1')(
        graphs_in, actions)
    critic_2 = RoboBalletCriticNet(config=self.config, name='critic_2')(
        graphs_in, actions)
    return critic_1, critic_2


def init_network_variables(
    planning_env_config: planning_env.PlanningEnvConfig,
    network: (RoboBalletPolicyNet | RoboBalletCriticNet |
              RoboBalletTwinCriticNet),
    seed: int,
    with_actions: bool = False,
) -> Any:
  """Initializes network variables (weights) for state-input networks."""
  # Build an environment just so we can get the graph tensor shapes to
  # initialize the network.
  planning_env_config = planning_env_config.replace(
      verify_targets_have_ik_solutions=False)
  shape_env = planning_env.PlanningEnv(config=planning_env_config)
  obs = shape_env.reset().observation
  obs_spec = shape_env.observation_spec()
  graph_features = compute_features.make_graph_features(
      obs, obs_spec, network.config.feature_config)
  if with_actions:
    actions = jnp.zeros(
        shape=(1,) + shape_env.action_spec().shape, dtype=jnp.float32
    )
    return network.init(jax.random.key(seed), graph_features, actions)
  else:
    return network.init(jax.random.key(seed), graph_features)


class GraphTopology(struct.PyTreeNode):
  """Constant data required to reconstruct graphs (or batch of graphs)."""

  senders: jnp.ndarray
  receivers: jnp.ndarray
  n_node: jnp.ndarray
  n_edge: jnp.ndarray

  # jax.jit requires static args to be hashable. Because we compile the train
  # step function ahead of time, we aren't actually hashing this every step (
  # when jax functions are compiled ahead of time (instead of automatically)
  # through calling jax.jit(fn)(...), it's the caller's responsibility to
  # recompile when static args change. We know the topology never changes, so
  # we just never recompile. Technically we can just return 0 here to trick
  # jax.jit into accepting a topology as a static arg. Nevertheless, we provide
  # a working __hash__ implementation here for correctness. Performance doesn't
  # matter.
  def __hash__(self):
    def hash_arr(arr):
      return hash(np.array(arr).tobytes())
    return hash((hash_arr(self.senders),
                 hash_arr(self.receivers),
                 hash_arr(self.n_node),
                 hash_arr(self.n_edge)))


def get_graph_topology(
    planning_env_config: planning_env.PlanningEnvConfig,
    feature_config: compute_features.FeatureConfig,
    batch_size: int = 1
) -> GraphTopology:
  """Returns graph topology for a batch of observations.

  Returns graph topology for a batch of `batch_size` observations from the
  specified env config. This allows us to batch graphs by simply concatenating
  the nodes, edges, and globals arrays, and reusing these connectivity data.

  Args:
    planning_env_config: Env config for observation shapes.
    feature_config: Config for computing features.
    batch_size: Batch size
  """

  planning_env_config = planning_env_config.replace(
      verify_targets_have_ik_solutions=False)
  env = planning_env.PlanningEnv(config=planning_env_config)
  obs = env.reset().observation
  obs_spec = env.observation_spec()
  example_graph = compute_features.make_graph_features(
      obs, obs_spec, feature_config)
  example_batch = jraph.batch([example_graph] * batch_size)
  return GraphTopology(
      senders=example_batch.senders,
      receivers=example_batch.receivers,
      n_node=example_batch.n_node,
      n_edge=example_batch.n_edge,
  )


def reconstruct_graph(
    nodes: np.ndarray,
    edges: np.ndarray,
    globals_: np.ndarray,
    topology: GraphTopology,
) -> jraph.GraphsTuple:
  """Reconstruct a GraphsTuple from nodes, edges, globals, and topology."""
  return jraph.GraphsTuple(
      nodes=jnp.asarray(nodes),
      edges=jnp.asarray(edges),
      globals=jnp.asarray(globals_),
      senders=topology.senders,
      receivers=topology.receivers,
      n_node=topology.n_node,
      n_edge=topology.n_edge,
  )


def _reshape_batch_dim_away(
    x: np.ndarray | jnp.ndarray,
) -> np.ndarray | jnp.ndarray:
  """Reshape (batch, first, ...) to (batch * first, ...)."""
  if isinstance(x, jnp.ndarray):
    reshape_fn = jnp.reshape
  else:
    reshape_fn = np.reshape
  return reshape_fn(x, (x.shape[0] * x.shape[1],) + x.shape[2:])


def reconstruct_graph_from_batched_data(
    batched_nodes: np.ndarray | jnp.ndarray,
    batched_edges: np.ndarray | jnp.ndarray,
    batched_globals: np.ndarray | jnp.ndarray,
    batched_topology: GraphTopology,
) -> jraph.GraphsTuple:
  """Reconstruct a GraphsTuple from batched nodes, edges, globals, and topology.

  Args:
    batched_nodes: Nodes in shape (num_graphs, n_nodes) + (nodes.shape)
    batched_edges: Edges in shape (num_graphs, n_edges) + (edges.shape)
    batched_globals: Globals in shape (num_graphs, n_globals) + (globals.shape)
    batched_topology: Batched topology from
      get_graph_topology(batch_size=num_graphs).

  Returns:
    Batched GraphsTuple.
  """
  # First we check for consistency.
  batch_size = batched_nodes.shape[0]
  assert batched_edges.shape[0] == batch_size
  assert batched_globals.shape[0] == batch_size
  assert batched_topology.n_node.shape[0] == batch_size
  assert batched_topology.n_edge.shape[0] == batch_size

  concat_nodes = _reshape_batch_dim_away(batched_nodes)
  concat_edges = _reshape_batch_dim_away(batched_edges)
  concat_globals = _reshape_batch_dim_away(batched_globals)
  assert concat_globals.shape[0] == batch_size

  return jraph.GraphsTuple(
      nodes=jnp.asarray(concat_nodes),
      edges=jnp.asarray(concat_edges),
      globals=jnp.asarray(concat_globals),
      senders=batched_topology.senders,
      receivers=batched_topology.receivers,
      n_node=batched_topology.n_node,
      n_edge=batched_topology.n_edge,
  )


def update_params_for_cpu_inference(params: Any) -> Any:
  """Updates params for CPU inference."""
  # Filter out the bf16 dtype.
  return jax.tree_util.tree_map(
      lambda x: x.astype(jnp.float32) if x.dtype == jnp.bfloat16 else x,
      params,
  )
