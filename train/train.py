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

"""Training library."""

from collections.abc import Callable
from concurrent import futures
import os
import pprint
import threading
import time
from typing import Any

from absl import logging
import chex
from flax import struct
import jax
from jax import numpy as jnp
import jraph
import launchpad as lp
import numpy as np
import optax
from optax import schedules
from orbax import checkpoint
import reverb
import tensorboardX

from agents import actor
from agents import compute_features
from agents import networks
from environments import planning_env
from train import stats as stats_lib


_DEBUG_MODE = False


# Timeout for jax dispatch.
_JAX_DISPATCH_TIMEOUT_S = 60


# In an exponential learning rate decay schedule, the number of steps per decay
# period is just for making the decay rate number easier to intuitively
# understand, and doesn't actually provide an additional control independent
# from decay rate. Since we are doing Vizier tuning, we just set the steps to a
# constant.
#
# decay_factor = decay_rate ** (step / steps_per_decay)
# any change to steps_per_decay is equivalent to a change in decay_rate:
# decay_factor = (decay_rate ** (1 / steps_per_decay)) ** step
_EXP_STEPS_PER_DECAY = 100000


ParamsType = chex.ArrayTree


def _exists(path: str) -> bool:
  """Returns True if the path exists, False otherwise."""
  return os.path.exists(path)


def _listdir(path: str) -> list[str]:
  """Returns a list of files in the path."""
  return os.listdir(path)


class TrainConfig(struct.PyTreeNode):
  """Training configs.

  Attributes:
    policy_learning_rate: Initial policy network learning rate.
    policy_learning_rate_decay_rate: Decay rate for policy learning rate.
    twin_critic_learning_rate: Initial critic network learning rate.
    twin_critic_learning_rate_decay_rate: Decay rate for critic learning rate.
    train_steps: Number of training steps.
    max_train_time_s: Maximum training time (for Vizier).
    train_batch_size: Batch size.
    checkpoint_interval: How often we are saving the network to CNS.
    stats_reporting_interval: How often we report training stats.
    fused_training_steps: How many training steps per jax call.
    seed: Network initialisation seed base.
    target_sigma: Target network noise scale.
    noise_clip: Target network noise clip range (before scaling by sigma).
    tau: Soft target network update constant.
    discount: Q-learning discount.
    reward_loss_weight: Weight of the reward aux loss.
  """

  policy_learning_rate: float = 5e-5
  policy_learning_rate_decay_rate: float = 0.98
  twin_critic_learning_rate: float = 5e-5
  twin_critic_learning_rate_decay_rate: float = 0.95
  train_steps: int = 20_000_000
  max_train_time_s: int | None = None
  train_batch_size: int = 8 if _DEBUG_MODE else 64
  checkpoint_interval: int = 10000
  stats_reporting_interval: int = 1000
  fused_training_steps: int = 20
  seed: int = 42

  # TD3 hyper-parameters.
  target_sigma: float = 5e-4
  noise_clip: float = 0.4
  policy_tau: float = 4e-5
  critic_tau: float = 8e-5
  discount: float = 0.94

  # Auxiliary loss weights.
  reward_loss_weight: float = 0.0


class TD3TrainState(struct.PyTreeNode):
  """TD3 training state."""

  step: jnp.ndarray
  policy_params: ParamsType
  twin_critic_params: ParamsType

  target_policy_params: ParamsType
  target_twin_critic_params: ParamsType

  policy_opt_state: optax.OptState
  twin_critic_opt_state: optax.OptState

  random_key: jax.Array


class TD3Optimisers(struct.PyTreeNode):
  """TD3 optimisers."""

  twin_critic_optimiser: optax.GradientTransformation
  policy_optimiser: optax.GradientTransformation


def _make_schedule(init_value: float, decay_rate: float) -> optax.Schedule:
  """Make a learning rate schedule."""
  return schedules.exponential_decay(
      init_value=init_value,
      transition_steps=_EXP_STEPS_PER_DECAY,
      decay_rate=decay_rate,
  )


def _make_optimisers(train_config: TrainConfig) -> TD3Optimisers:
  """Creates TD3 optimisers."""
  policy_schedule = _make_schedule(
      init_value=train_config.policy_learning_rate,
      decay_rate=train_config.policy_learning_rate_decay_rate,
  )
  twin_critic_schedule = _make_schedule(
      init_value=train_config.twin_critic_learning_rate,
      decay_rate=train_config.twin_critic_learning_rate_decay_rate,
  )
  return TD3Optimisers(
      policy_optimiser=optax.adam(learning_rate=policy_schedule),
      twin_critic_optimiser=optax.adam(learning_rate=twin_critic_schedule),
  )


def _debug_print(**kwargs):  # pylint: disable=unused-private-name
  if _DEBUG_MODE:
    for k, v in kwargs.items():
      jax.debug.print(f'{k} = ' + '{v}', v=v, ordered=True)


def _shard_arrays(
    args: dict[str, jnp.ndarray], sharding: jax.sharding.Sharding | jax.Device
) -> dict[str, jnp.ndarray]:
  return {
      k: jax.lax.with_sharding_constraint(v, sharding) for k, v in args.items()
  }


def _debug_print_str(s):  # pylint: disable=unused-private-name
  if _DEBUG_MODE:
    jax.debug.print(s, ordered=True)


# jax.grad requires differentiating targets to be a positional argument, so we
# have to pass that in separately.
def _critic_loss(
    twin_critic_params: ParamsType,
    train_state: TD3TrainState,
    train_config: TrainConfig,
    observations: jraph.GraphsTuple,
    next_observations: jraph.GraphsTuple,
    actions: jnp.ndarray,
    rewards: jnp.ndarray,
    terminals: jnp.ndarray,
    policy_apply_fn: Callable[..., Any],
    twin_critic_apply_fn: Callable[..., Any],
    rng_key: jax.Array,
) -> jnp.ndarray:
  """TD3 critic loss."""

  # See: https://arxiv.org/pdf/1802.09477.pdf

  critic1_tm1_outputs, critic2_tm1_outputs = twin_critic_apply_fn(
      twin_critic_params, observations, actions
  )
  q1_tm1 = critic1_tm1_outputs.q_values
  r1_tm1 = critic1_tm1_outputs.rewards
  q2_tm1 = critic2_tm1_outputs.q_values
  r2_tm1 = critic2_tm1_outputs.rewards

  new_action = policy_apply_fn(
      train_state.target_policy_params, next_observations
  )
  scaled_clipped_noise = train_config.target_sigma * jnp.clip(
      jax.random.normal(key=rng_key, shape=new_action.shape),
      -train_config.noise_clip,
      train_config.noise_clip,
  )
  noisy_action = jnp.clip(new_action + scaled_clipped_noise, -1.0, 1.0)

  critic1_outputs, critic2_outputs = twin_critic_apply_fn(
      train_state.target_twin_critic_params, next_observations, noisy_action
  )
  q1_t = critic1_outputs.q_values
  q2_t = critic2_outputs.q_values
  not_terminals = 1 - terminals
  future_rewards = train_config.discount * jnp.minimum(q1_t, q2_t)
  target = jax.lax.stop_gradient(rewards + not_terminals * future_rewards)

  td_loss = jnp.mean(jnp.square(target - q1_tm1)) + jnp.mean(
      jnp.square(target - q2_tm1)
  )
  reward_loss = jnp.mean(jnp.square(rewards - r1_tm1)) + jnp.mean(
      jnp.square(rewards - r2_tm1)
  )

  return td_loss + reward_loss * train_config.reward_loss_weight


# The policy update happens after the critic update, and need to use the
# updated critic params.
def _policy_loss(
    policy_params: ParamsType,
    twin_critic_params: ParamsType,
    observations: jraph.GraphsTuple,
    policy_apply_fn: Callable[..., Any],
    twin_critic_apply_fn: Callable[..., Any],
) -> jnp.ndarray:
  """TD3 policy loss."""
  action = policy_apply_fn(policy_params, observations)
  # Policy loss only uses one of the critics.
  critic1_outs, _ = twin_critic_apply_fn(
      twin_critic_params, observations, action
  )
  return -jnp.mean(critic1_outs.q_values)


def _train_step_impl(
    train_state: TD3TrainState,
    batched_args: dict[str, jnp.ndarray],
    batched_topology: networks.GraphTopology,
    policy_apply_fn: Callable[..., Any],
    twin_critic_apply_fn: Callable[..., Any],
    optimisers: TD3Optimisers,
    train_config: TrainConfig,
) -> tuple[TD3TrainState, dict[str, jnp.ndarray]]:
  """Compilable train step function implementation."""

  twin_critic_rng_key, key_next_step = jax.random.split(
      train_state.random_key, 2
  )

  batched_states = networks.reconstruct_graph_from_batched_data(
      batched_args['nodes'],
      batched_args['edges'],
      batched_args['globals'],
      batched_topology,
  )
  batched_next_states = networks.reconstruct_graph_from_batched_data(
      batched_args['next_nodes'],
      batched_args['next_edges'],
      batched_args['next_globals'],
      batched_topology,
  )

  # First we update the twin critics.
  twin_critic_loss_and_grad = jax.value_and_grad(_critic_loss)
  twin_critic_loss, twin_critic_grad = twin_critic_loss_and_grad(
      train_state.twin_critic_params,
      train_state=train_state,
      train_config=train_config,
      observations=batched_states,
      next_observations=batched_next_states,
      actions=batched_args['actions'],
      rewards=batched_args['rewards'],
      terminals=batched_args['terminals'],
      policy_apply_fn=policy_apply_fn,
      twin_critic_apply_fn=twin_critic_apply_fn,
      rng_key=twin_critic_rng_key,
  )

  twin_critic_updates, twin_critic_opt_state = (
      optimisers.twin_critic_optimiser.update(
          twin_critic_grad, train_state.twin_critic_opt_state
      )
  )
  twin_critic_params = optax.apply_updates(
      train_state.twin_critic_params, twin_critic_updates
  )

  target_twin_critic_params = optax.incremental_update(
      new_tensors=twin_critic_params,
      old_tensors=train_state.target_twin_critic_params,
      step_size=train_config.critic_tau,
  )

  policy_loss_and_grad = jax.value_and_grad(_policy_loss)
  policy_loss, policy_grad = policy_loss_and_grad(
      train_state.policy_params,
      twin_critic_params=twin_critic_params,
      observations=batched_states,
      policy_apply_fn=policy_apply_fn,
      twin_critic_apply_fn=twin_critic_apply_fn,
  )

  policy_updates, policy_opt_state = optimisers.policy_optimiser.update(
      policy_grad, train_state.policy_opt_state
  )
  policy_params = optax.apply_updates(train_state.policy_params, policy_updates)
  target_policy_params = optax.incremental_update(
      new_tensors=policy_params,
      old_tensors=train_state.target_policy_params,
      step_size=train_config.policy_tau,
  )

  stats = {
      'policy_loss': policy_loss,
      'twin_critic_loss': twin_critic_loss,
  }
  return (
      TD3TrainState(
          step=train_state.step + 1,
          policy_params=policy_params,
          twin_critic_params=twin_critic_params,
          target_policy_params=target_policy_params,
          target_twin_critic_params=target_twin_critic_params,
          policy_opt_state=policy_opt_state,
          twin_critic_opt_state=twin_critic_opt_state,
          random_key=key_next_step,
      ),
      stats,
  )


def _fused_train_step_impl(
    num_fused_steps: int,
    train_state: TD3TrainState,
    batched_args: dict[str, jnp.ndarray],
    batched_topology: networks.GraphTopology,
    policy_apply_fn: Callable[..., Any],
    twin_critic_apply_fn: Callable[..., Any],
    optimisers: TD3Optimisers,
    train_config: TrainConfig,
) -> tuple[TD3TrainState, dict[str, jnp.ndarray]]:
  """Runs `num_fused_steps` training steps as one Jax operation."""

  def single_train_step(step_i, train_state_and_stats):
    old_train_state, old_stats = train_state_and_stats
    start_batch_idx = step_i * train_config.train_batch_size

    def get_batch_slice(x):
      return jax.lax.dynamic_slice_in_dim(
          operand=x,
          start_index=start_batch_idx,
          slice_size=train_config.train_batch_size,
      )

    def get_batch_slice_dict(x: dict[str, jnp.ndarray]):
      return {k: get_batch_slice(v) for k, v in x.items()}

    new_train_state, new_stats = _train_step_impl(
        train_state=old_train_state,
        batched_args=get_batch_slice_dict(batched_args),
        batched_topology=batched_topology,
        policy_apply_fn=policy_apply_fn,
        twin_critic_apply_fn=twin_critic_apply_fn,
        optimisers=optimisers,
        train_config=train_config,
    )
    combined_stats = {k: v + new_stats[k] for k, v in old_stats.items()}
    return (new_train_state, combined_stats)

  init_stats = {
      'policy_loss': jnp.zeros([], dtype=jnp.float32),
      'twin_critic_loss': jnp.zeros([], dtype=jnp.float32),
  }
  return jax.lax.fori_loop(
      lower=0,
      upper=num_fused_steps,
      body_fun=single_train_step,
      init_val=(train_state, init_stats),
  )


def _compile_step_impl_fn(
    train_config: TrainConfig,
    train_state: TD3TrainState,
    batched_args: dict[str, jnp.ndarray],
    batched_topology: networks.GraphTopology,
    policy_net: networks.RoboBalletPolicyNet,
    twin_critic_net: networks.RoboBalletTwinCriticNet,
    optimisers: TD3Optimisers,
) -> Callable[..., Any]:
  """Compiles the fused training step fn."""

  args = dict(
      num_fused_steps=train_config.fused_training_steps,
      train_state=train_state,
      batched_args=batched_args,
      batched_topology=batched_topology,
      policy_apply_fn=policy_net.apply,
      twin_critic_apply_fn=twin_critic_net.apply,
      optimisers=optimisers,
      train_config=train_config,
  )

  static_argnames = [
      'num_fused_steps',
      'batched_topology',
      'policy_apply_fn',
      'twin_critic_apply_fn',
      'optimisers',
      'train_config',
  ]
  jit_fn = jax.jit(_fused_train_step_impl, static_argnames=static_argnames)
  lowered = jit_fn.lower(**args)
  lowered_text = lowered.as_text()
  logging.info('Compiling train step fn (%d lines): ', lowered_text.count('\n'))

  # Log HLO to stdout (too big for logging).
  print('Step fn HLO:')
  print(lowered_text)

  compiled = lowered.compile()
  logging.info('Compilation done')
  logging.info('Memory: %s', str(compiled.memory_analysis()))

  return compiled


def _fused_train_step(
    compiled_step_impl_fn: Callable[..., Any],
    step: int,
    train_state: TD3TrainState,
    batched_args: dict[str, jnp.ndarray],
) -> tuple[TD3TrainState, dict[str, jnp.ndarray]]:
  """Performs a single training step."""
  args = dict(
      train_state=train_state,
      batched_args=batched_args,
  )

  with jax.profiler.StepTraceAnnotation('train', step_num=step):
    new_state, stats_dict = compiled_step_impl_fn(**args)

  return new_state, stats_dict


def _make_train_state(
    planning_env_config: planning_env.PlanningEnvConfig,
    policy_net: networks.RoboBalletPolicyNet,
    twin_critic_net: networks.RoboBalletTwinCriticNet,
    train_config: TrainConfig,
) -> TD3TrainState:
  """Creates a train state."""

  # In TD3 we have two network architectures (policy and critic) and 6
  # sets of params -
  #
  # - policy
  # - critic x2
  # - policy target
  # - critic target x 2

  # The 3 main networks are all independently parameterised, and the target
  # networks are initialised to be the same as the corresponding active network.

  rng = np.random.default_rng(train_config.seed)

  def rand_int32() -> int:
    return rng.integers(low=0, high=2**32 - 1)

  policy_params = networks.init_network_variables(
      planning_env_config=planning_env_config,
      network=policy_net,
      seed=rand_int32(),
  )
  twin_critic_params = networks.init_network_variables(
      planning_env_config=planning_env_config,
      network=twin_critic_net,
      seed=rand_int32(),
      with_actions=True,
  )

  optimisers = _make_optimisers(train_config=train_config)

  return TD3TrainState(
      step=jnp.array(0, dtype=jnp.uint32),
      policy_params=policy_params,
      twin_critic_params=twin_critic_params,
      # We can copy by reference here because Jax arrays are immutable and we
      # will create new ones when we update anyways.
      target_policy_params=policy_params,
      target_twin_critic_params=twin_critic_params,
      policy_opt_state=optimisers.policy_optimiser.init(policy_params),
      twin_critic_opt_state=optimisers.twin_critic_optimiser.init(
          twin_critic_params
      ),
      random_key=jax.random.PRNGKey(rand_int32()),
  )


def make_example_checkpoint_tree(
    planning_env_config: planning_env.PlanningEnvConfig,
    policy_net: networks.RoboBalletPolicyNet,
    twin_critic_net: networks.RoboBalletTwinCriticNet,
    train_config: TrainConfig,
) -> dict[str, Any]:
  """Make an example checkpoint tree for restoring from Orbax."""
  return {
      'step': 0,
      'train_state': _make_train_state(
          planning_env_config=planning_env_config,
          policy_net=policy_net,
          twin_critic_net=twin_critic_net,
          train_config=train_config,
      ),
      'elapsed_time_ns': 0,
  }


def check_not_none(a: Any) -> Any:
  assert a is not None
  return a


class LearnerNode:
  """Learner LaunchPad node."""

  def __init__(
      self,
      train_config: TrainConfig,
      feature_config: compute_features.FeatureConfig,
      policy_net: networks.RoboBalletPolicyNet,
      twin_critic_net: networks.RoboBalletTwinCriticNet,
      planning_env_config: planning_env.PlanningEnvConfig,
      learner_id: int,
      reverb_client: reverb.Client,
      checkpoint_dir: str | None,
      starting_checkpoint_dir: str | None,
      tensorboard_log_dir: str | None,
  ):
    # This is for determining when to stop Vizier trials. We set the start time
    # when the first step is done, so we are less affected by initialisation
    # delays etc skewing results.
    self._experiment_start_time = None

    self._train_config = train_config
    self._feature_config = feature_config
    self._policy_net = policy_net
    self._twin_critic_net = twin_critic_net
    self._learner_id = learner_id
    self._reverb_client = reverb_client
    self._checkpoint_dir = checkpoint_dir
    self._checkpointer = checkpoint.AsyncCheckpointer(
        checkpoint.PyTreeCheckpointHandler()
    )
    self._step = 0

    # We only drop out of Jax every `fused_training_steps` steps, so everything
    # we need to do in Python must fall on multiples of those steps.
    assert (
        self._train_config.checkpoint_interval
        % self._train_config.fused_training_steps
        == 0
    )
    assert (
        self._train_config.stats_reporting_interval
        % self._train_config.fused_training_steps
        == 0
    )

    logging.info('TensorBoard log dir: %s', tensorboard_log_dir)
    if tensorboard_log_dir:
      self._summary_writer = tensorboardX.SummaryWriter(
          logdir=tensorboard_log_dir,
          filename_suffix=f'.learner.{learner_id}')
    else:
      self._summary_writer = None

    self._train_state = _make_train_state(
        planning_env_config=planning_env_config,
        policy_net=self._policy_net,
        twin_critic_net=self._twin_critic_net,
        train_config=self._train_config,
    )
    self._batch_graph_topology = networks.get_graph_topology(
        planning_env_config=planning_env_config,
        feature_config=self._feature_config,
        batch_size=self._train_config.train_batch_size,
    )
    self._single_graph_topology = networks.get_graph_topology(
        planning_env_config=planning_env_config,
        feature_config=self._feature_config,
        batch_size=1,
    )

    restore_path = None
    # First we check if we already have a checkpoint.
    # If we do, this takes priority over starting checkpoint_dir.
    if self._checkpoint_dir and _exists(self._checkpoint_dir):
      logging.info(
          'Checkpoint dir already exists: %s. Restoring latest checkpoint.',
          self._checkpoint_dir,
      )
      snapshot_dirs_list = _listdir(self._checkpoint_dir)
      snapshot_list = []
      for snapshot_dir in snapshot_dirs_list:
        try:
          snapshot_list.append(int(snapshot_dir))
        except ValueError:
          pass
      if snapshot_list:
        step = sorted(snapshot_list, reverse=True)[0]
        restore_path = os.path.join(self._checkpoint_dir, str(step))
    elif starting_checkpoint_dir:
      logging.info('Starting from checkpoint: %s', starting_checkpoint_dir)
      restore_path = starting_checkpoint_dir

    if restore_path:
      # If we are restoring from a checkpoint, make sure we have the right
      # sharding information.
      example_tree = make_example_checkpoint_tree(
          planning_env_config=planning_env_config,
          policy_net=policy_net,
          twin_critic_net=twin_critic_net,
          train_config=self._train_config,
      )
      restore_args = jax.tree_util.tree_map(
          lambda _: checkpoint.ArrayRestoreArgs(),
          example_tree,
      )
      restored = self._checkpointer.restore(
          restore_path,
          item={
              'step': 0,
              'train_state': self._train_state,
              'elapsed_time_ns': 0,
          },
          restore_args=restore_args,
      )
      self._step = restored['step']

      # We keep track of elapsed training time using a
      # start time, and checkpoint the elapsed time each time a checkpoint is
      # saved. On restore, we restore the elapsed time by creating a fake
      # start time that is 'elapsed time'-ago from the time now.
      self._experiment_start_time = time.time_ns() - int(
          restored['elapsed_time_ns']
      )

      self._train_state = restored['train_state']

    # We maintain an extra copy of the policy params so actors can fetch latest
    # policy params at any time, without having to synchronise with training
    # steps.
    self._policy_params = self._train_state.policy_params
    self._policy_params_step = self._step
    self._policy_params_lock = threading.Lock()

    self._last_train_stats_dict = None

  def get_policy_variables(self) -> tuple[ParamsType, int]:
    with self._policy_params_lock:
      logging.info('Client fetched network step: %d', self._policy_params_step)
      return self._policy_params, self._policy_params_step

  def _should_terminate_on_time(self) -> bool:
    if (
        self._experiment_start_time is None
        or self._train_config.max_train_time_s is None
    ):
      return False
    elapsed_time = (time.time_ns() - self._experiment_start_time) / 1e9
    return elapsed_time >= self._train_config.max_train_time_s

  def run(self):
    """Entry point for the trainer LaunchPad node."""
    logging.info('Learner %d started', self._learner_id)

    megabatch_size = (
        self._train_config.train_batch_size
        * self._train_config.fused_training_steps
    )

    ds = reverb.TrajectoryDataset.from_table_signature(
        server_address=self._reverb_client.server_address,
        table=actor.REPLAY_TABLE_NAME,
        max_in_flight_samples_per_worker=3 * megabatch_size,
    )
    ds_iter = ds.batch(megabatch_size).prefetch(3)
    ds_iter = ds_iter.as_numpy_iterator()

    train_stats = []
    trajectory_stats = []

    # We discard first report because the first report contains huge outliers
    # and distort curves, and the data is not often useful.
    first_report_discarded = False

    optimisers = _make_optimisers(train_config=self._train_config)

    logging.info('Learner %d: Waiting for first sample', self._learner_id)

    jax_dispatch_executor = futures.ThreadPoolExecutor(max_workers=1)

    train_step_outputs_future = None

    compiled_step_fn = None
    while (
        not lp.stop_event().is_set()
        and not self._should_terminate_on_time()
        and not self._step >= self._train_config.train_steps
    ):
      step_start_time = time.time_ns()
      sample = next(ds_iter).data
      if self._step == 0:
        logging.info('Learner %d: Got first megabatch', self._learner_id)

      obs = sample['observation']
      obs_next = sample['observation_next']
      state_nodes = obs['nodes']
      state_edges = obs['edges']
      state_globals = obs['globals']
      next_state_nodes = obs_next['nodes']
      next_state_edges = obs_next['edges']
      next_state_globals = obs_next['globals']
      batched_actions = sample['action']
      batched_rewards = sample['reward']
      batched_terminals = sample['terminal']

      if 'trajectory_stats' in sample:
        stats = sample['trajectory_stats']
        stats = {k: np.mean(v) for k, v in stats.items()}
        trajectory_stats.append(stats_lib.TrajectoryStats(**stats))

      sampling_duration_ms = (time.time_ns() - step_start_time) / 1e6

      if lp.stop_event().is_set():
        logging.info('Learner %d: Received stop signal', self._learner_id)
        break

      # If this is not the first step, this is when we update the train_state
      # and stats dict with the result of the last step. This is the latest
      # point we have to do this, before submitting the next step which needs
      # the output of the last step. Note that at this point the train_state
      # may not have actually materialised yet, only that we have a set of Jax
      # arrays (which means the operations have been dispatched). That is all
      # we need to pass the train state to the next step.
      # We are doing this last minute so that we can do all the CPU work between
      # steps (sampling new transitions, batching, etc) while Jax does the
      # dispatching in parallel.
      if train_step_outputs_future is not None:
        if self._step <= 1:
          logging.info('Learner %d: Waiting for dispatch future')
        try:
          self._train_state, self._last_train_stats_dict = (
              train_step_outputs_future.result(timeout=_JAX_DISPATCH_TIMEOUT_S)
          )
        except TimeoutError:
          logging.error('Learner %d: Timeout while waiting for Jax dispatch')
          raise
        with self._policy_params_lock:
          self._policy_params = self._train_state.policy_params
          self._policy_params_step = self._step
        if self._step <= 1:
          logging.info('Learner %d: Dispatch done')

      batched_args = dict(
          nodes=state_nodes,
          edges=state_edges,
          globals=state_globals,
          next_nodes=next_state_nodes,
          next_edges=next_state_edges,
          next_globals=next_state_globals,
          actions=batched_actions,
          rewards=batched_rewards,
          terminals=batched_terminals,
      )

      step_args = dict(
          train_config=self._train_config,
          train_state=self._train_state,
          batched_args=batched_args,
          batched_topology=self._batch_graph_topology,
          policy_net=self._policy_net,
          twin_critic_net=self._twin_critic_net,
          optimisers=optimisers,
      )

      if compiled_step_fn is None:
        compiled_step_fn = _compile_step_impl_fn(**step_args)

      # We are running a compiled function so we don't need to pass the static
      # arguments (they have been compiled in).
      del step_args['train_config']
      del step_args['batched_topology']
      del step_args['policy_net']
      del step_args['twin_critic_net']
      del step_args['optimisers']

      if self._step == 0:
        logging.info('Learner %d: Submitting to dispatch executor')
      train_step_outputs_future = jax_dispatch_executor.submit(
          _fused_train_step,
          compiled_step_impl_fn=compiled_step_fn,
          step=self._step,
          **step_args,
      )

      # This is using the stats from the last step (not the one that has just
      # been submitted). This avoids forcing a sync with the GPU, and allows the
      # CPU to run ahead.
      if self._last_train_stats_dict is not None:
        stats_dict = {
            k: float(np.asarray(v))
            for k, v in self._last_train_stats_dict.items()
        }
        stats = stats_lib.TrainingStats(
            **stats_dict,
        )
        stats.sampling_wall_time_ms = (
            sampling_duration_ms / self._train_config.fused_training_steps
        )
        stats.total_step_time_ms = (
            (time.time_ns() - step_start_time)
            / 1e6
            / self._train_config.fused_training_steps
        )
        stats.trajectory_network_age = (
            (self._step - trajectory_stats[-1].network_step)
            if trajectory_stats
            else 0
        )
        train_stats.append(stats)

      if lp.stop_event().is_set():
        logging.info('Learner %d: Received stop signal', self._learner_id)
      else:
        if (
            self._step % self._train_config.stats_reporting_interval == 0
            and self._step > 0
        ):
          if first_report_discarded:
            aggregated_train_stats = stats_lib.mean_aggregate_stats(train_stats)
            train_config = self._train_config
            policy_lr = float(
                _make_schedule(
                    init_value=train_config.policy_learning_rate,
                    decay_rate=train_config.policy_learning_rate_decay_rate,
                )(self._step)
            )
            twin_critic_lr = float(
                _make_schedule(
                    init_value=train_config.twin_critic_learning_rate,
                    decay_rate=train_config.twin_critic_learning_rate_decay_rate,
                )(self._step)
            )
            aggregated_train_stats['policy_lr'] = policy_lr
            aggregated_train_stats['twin_critic_lr'] = twin_critic_lr
            all_stats = {
                'step': self._step,
                'train': aggregated_train_stats,
            }
            if trajectory_stats:
              all_stats['acting'] = stats_lib.mean_aggregate_stats(
                  trajectory_stats
              )
            logging.info('%s', pprint.pformat(all_stats))

            # Write to TensorBoard.
            if self._summary_writer is not None and self._step > 0:
              for field_name, value in aggregated_train_stats.items():
                self._summary_writer.add_scalar('train/' + field_name, value,
                                                global_step=self._step)
              if 'acting' in all_stats:
                for field_name, value in all_stats['acting'].items():
                  self._summary_writer.add_scalar('act/' + field_name, value,
                                                  global_step=self._step)
              self._summary_writer.flush()
          else:
            first_report_discarded = True

          train_stats = []
          trajectory_stats = []

        # Write a checkpoint
        if (
            self._checkpoint_dir is not None
            and self._step % self._train_config.checkpoint_interval == 0
        ):
          checkpoint_path = f'{self._checkpoint_dir}/{self._step}'
          if _exists(checkpoint_path):
            logging.warn('Checkpoint path %s already exists. Skipping')
          else:
            logging.info('Checkpointing to: %s', checkpoint_path)
            elapsed_time = 0
            if self._experiment_start_time is not None:
              elapsed_time = time.time_ns() - self._experiment_start_time
            ckpt = {
                'step': self._step,
                'elapsed_time_ns': elapsed_time,
                'train_state': self._train_state,
            }
            save_args = jax.tree_util.tree_map(
                lambda _: checkpoint.SaveArgs(aggregate=False),
                ckpt,
            )
            self._checkpointer.save(checkpoint_path, ckpt, save_args=save_args)
      self._step += self._train_config.fused_training_steps
      if self._experiment_start_time is None:
        self._experiment_start_time = time.time_ns()

    if self._step >= self._train_config.train_steps:
      logging.info('Step count limit reached. Terminating experiment.')
      lp.stop()
    elif self._should_terminate_on_time():
      logging.info('Time limit reached, Terminating experiment.')
      lp.stop()
    elif lp.stop_event().is_set():
      logging.info('Experiment terminated externally.')
