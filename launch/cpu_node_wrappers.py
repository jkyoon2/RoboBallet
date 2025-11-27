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

"""CPU-only node wrappers for Actor and Evaluator.

CRITICAL: Environment variables are set at MODULE LEVEL, before any other
imports. This ensures JAX is configured for CPU-only before it can be
initialized by any unpickling operations.
"""

# =============================================================================
# CRITICAL: Set environment variables FIRST, before ANY other imports!
# This must happen at module load time, before unpickling triggers JAX init.
# =============================================================================
import os
import sys

# Check if JAX has already been initialized in this process
# If not, configure it for CPU-only
if 'jax' not in sys.modules:
    os.environ['JAX_PLATFORMS'] = 'cpu'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=1'
    # Prevent JAX from pre-allocating GPU memory (fallback if GPU is still visible)
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.0'

# =============================================================================
# Now it's safe to define our wrapper classes
# =============================================================================


def _configure_jax_for_cpu():
    """Additional JAX configuration for CPU. Called in __init__ as a safeguard."""
    # Double-check environment is set (in case of any edge cases)
    os.environ['JAX_PLATFORMS'] = 'cpu'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    # Try to configure JAX if it hasn't been configured yet
    try:
        import jax
        # This will only work if JAX hasn't been initialized yet
        jax.config.update('jax_platforms', 'cpu')
    except Exception:
        pass  # JAX might already be initialized, which is fine if env vars were set


class CPUActorNode:
    """Wrapper for ActorNode that runs on CPU only.
    
    When instantiated in a Launchpad subprocess, this wrapper:
    1. Environment variables are already set at module import time
    2. Calls the network factory to create the network (JAX initialized as CPU)
    3. Creates and delegates to the actual ActorNode
    """

    def __init__(
        self,
        actor_config,
        env_config,
        network_factory,
        actor_id,
        reverb_client,
        learner_client,
    ):
        # Extra safeguard: configure JAX for CPU
        _configure_jax_for_cpu()
        
        # Now it's safe to call the factory - JAX should be CPU-only
        network = network_factory()
        
        # Import the actual actor module
        from agents import actor
        
        # Create the actual ActorNode
        self._node = actor.ActorNode(
            actor_config=actor_config,
            env_config=env_config,
            network=network,
            actor_id=actor_id,
            reverb_client=reverb_client,
            learner_client=learner_client,
        )

    def run(self):
        """Delegate to the actual ActorNode's run method."""
        return self._node.run()


class CPUEvaluatorNode:
    """Wrapper for EvaluatorNode that runs on CPU only.
    
    When instantiated in a Launchpad subprocess, this wrapper:
    1. Environment variables are already set at module import time
    2. Calls the network factory to create the network (JAX initialized as CPU)
    3. Creates and delegates to the actual EvaluatorNode
    """

    def __init__(
        self,
        evaluator_config,
        actor_config,
        env_config,
        network_factory,
        evaluator_name,
        learner_client,
        tensorboard_log_dir,
    ):
        # Extra safeguard: configure JAX for CPU
        _configure_jax_for_cpu()
        
        # Now it's safe to call the factory - JAX should be CPU-only
        network = network_factory()
        
        # Import the actual evaluator module
        from agents import evaluator
        
        # Create the actual EvaluatorNode
        self._node = evaluator.EvaluatorNode(
            evaluator_config=evaluator_config,
            actor_config=actor_config,
            env_config=env_config,
            network=network,
            evaluator_name=evaluator_name,
            learner_client=learner_client,
            tensorboard_log_dir=tensorboard_log_dir,
        )

    def run(self):
        """Delegate to the actual EvaluatorNode's run method."""
        return self._node.run()
