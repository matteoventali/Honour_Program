"""Gymnasium wrapper that augments observations with an LTLf DFA state."""

import gymnasium as gym
import numpy as np

from utils import phi_mapping_sequential


# =============================================================================
# LTLf observation and reward wrapper
# =============================================================================

class LTLfShapingWrapper(gym.Wrapper):
    """Apply the synthetic task reward and potential-based reward shaping."""

    def __init__(
        self,
        env,
        abstract_mdp,
        metrics,
        use_shaping=True,
        shaping_scale=1.0,
        goal_reward=10000.0,
    ):
        super().__init__(env)
        self.abstract_mdp = abstract_mdp
        self.metrics = metrics
        self.use_shaping = use_shaping
        self.shaping_scale = shaping_scale
        self.goal_reward = goal_reward
        self.automaton_states = list(abstract_mdp.automaton.states)
        self.state_to_index = {
            state: index for index, state in enumerate(self.automaton_states)
        }
        self.current_dfa_state = None
        self.previous_observation = None
        self.episode_task_reward = 0.0
        self.episode_total_reward = 0.0
        self.episode_length = 0

        # The policy receives the physical LunarLander state followed by a
        # one-hot encoding of the active DFA state.
        observation_size = env.observation_space.shape[0] + len(self.automaton_states)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_size,),
            dtype=np.float32,
        )

    @staticmethod
    def _abstract_position(observation):
        x, y, _ = phi_mapping_sequential(observation, 0)
        return x, y

    def _augment_observation(self, observation, dfa_state):
        one_hot = np.zeros(len(self.automaton_states), dtype=np.float32)
        one_hot[self.state_to_index[dfa_state]] = 1.0
        return np.concatenate((observation, one_hot)).astype(np.float32)

    def reset(self, **kwargs):
        """Reset the environment and evaluate the initial observation in the DFA."""
        observation, info = self.env.reset(**kwargs)
        x, y = self._abstract_position(observation)
        valuation = self.abstract_mdp._get_truth_assignment(x, y)

        # The DFA initial node represents the empty trace. Consuming s0 here
        # ensures that the first policy observation carries the correct state.
        pre_trace_state = self.abstract_mdp.automaton.get_initial_q()
        self.current_dfa_state = self.abstract_mdp.automaton.get_next_q(
            pre_trace_state,
            valuation,
        )
        self.previous_observation = observation
        self.episode_task_reward = 0.0
        self.episode_total_reward = 0.0
        self.episode_length = 0
        return self._augment_observation(observation, self.current_dfa_state), info

    def step(self, action):
        """Advance the environment, DFA and potential-based reward process."""
        # The native LunarLander reward is intentionally discarded: this
        # experiment learns exclusively from the temporal task and shaping.
        observation, _, terminated, truncated, info = self.env.step(action)
        environment_terminated = terminated
        self.episode_length += 1

        x, y = self._abstract_position(self.previous_observation)
        next_x, next_y = self._abstract_position(observation)
        valuation = self.abstract_mdp._get_truth_assignment(next_x, next_y)
        next_dfa_state = self.abstract_mdp.automaton.get_next_q(
            self.current_dfa_state,
            valuation,
        )

        task_reward = 0.0
        task_success = False

        # Entering an accepting DFA state completes the temporal task and ends
        # the Gymnasium episode with the configured synthetic goal reward.
        if next_dfa_state != self.current_dfa_state:
            self.metrics.record_transition(self.current_dfa_state, next_dfa_state)
            if self.abstract_mdp.automaton.is_goal_reached(next_dfa_state):
                task_reward = self.goal_reward
                task_success = True
                terminated = True

        abstract_state = (x, y, self.current_dfa_state)
        next_abstract_state = (next_x, next_y, next_dfa_state)
        shaping_reward = 0.0

        # Potential-based shaping is applied only when the complete abstract
        # state changes, using F(s,s') = K * (gamma * V*(s') - V*(s)).
        if self.use_shaping and abstract_state != next_abstract_state:
            current_potential = self.abstract_mdp.v_star.get(abstract_state, 0.0)
            next_potential = self.abstract_mdp.v_star.get(next_abstract_state, 0.0)
            shaping_reward = self.shaping_scale * (
                self.abstract_mdp.gamma * next_potential - current_potential
            )

        total_reward = task_reward + shaping_reward
        self.episode_task_reward += task_reward
        self.episode_total_reward += total_reward

        # Store one coherent record only when an episode has actually ended.
        if terminated or truncated:
            if task_success:
                end_reason = "success"
            elif environment_terminated:
                end_reason = "environment_terminated"
            else:
                end_reason = "truncated"
            self.metrics.record_episode(
                self.episode_task_reward,
                self.episode_total_reward,
                self.episode_length,
                end_reason,
                self.use_shaping,
            )

        self.current_dfa_state = next_dfa_state
        self.previous_observation = observation
        augmented_observation = self._augment_observation(
            observation,
            next_dfa_state,
        )
        return augmented_observation, total_reward, terminated, truncated, info
