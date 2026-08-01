"""Gymnasium wrappers used by the SB3 SAC experiment."""

from collections import Counter

import gymnasium as gym
import numpy as np

from utils import phi_mapping_sequential


class DiscreteToContinuousActionWrapper(gym.ActionWrapper):
    """Expose a one-dimensional continuous action to a continuous-control agent.

    LunarLander still receives one of its original discrete actions.  A value
    selected by SAC is clipped to ``[-1, 1]``, scaled to ``[0, n)`` and floored.
    Clipping the final index is important because the upper Box endpoint maps
    exactly to ``n`` before flooring.
    """

    def __init__(self, env):
        super().__init__(env)
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise TypeError("The wrapped environment must have a Discrete action space")

        self.discrete_action_space = env.action_space
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def action(self, action):
        """Convert SAC's continuous action into an original LunarLander action."""
        continuous_action = np.asarray(action, dtype=np.float32)
        if continuous_action.size != 1:
            raise ValueError(
                "Expected one continuous action value, "
                f"received shape {continuous_action.shape}"
            )

        scalar_action = float(continuous_action.reshape(-1)[0])
        if not np.isfinite(scalar_action):
            raise ValueError("The continuous action must be finite")

        clipped_action = float(np.clip(scalar_action, -1.0, 1.0))
        unit_action = (clipped_action + 1.0) / 2.0
        discrete_action = int(np.floor(unit_action * self.discrete_action_space.n))
        return int(np.clip(discrete_action, 0, self.discrete_action_space.n - 1))


class LTLfTaskWrapper(gym.Wrapper):
    """Add the DFA state, synthetic task reward and potential-based shaping.

    This wrapper contains only environment semantics.  SAC construction,
    optimization, callbacks and checkpointing intentionally remain in
    ``trainer.py``.
    """

    def __init__(self, env, abstract_mdp, use_shaping=True, shaping_scale=1.0, goal_reward=10000.0):
        super().__init__(env)
        self.abstract_mdp = abstract_mdp
        self.automaton = abstract_mdp.automaton
        self.use_shaping = bool(use_shaping)
        self.shaping_scale = float(shaping_scale)
        self.goal_reward = float(goal_reward)
        self.automaton_states = list(self.automaton.states)
        self.state_to_index = {
            q: index for index, q in enumerate(self.automaton_states)
        }

        if not isinstance(env.observation_space, gym.spaces.Box):
            raise TypeError("The wrapped environment must have a Box observation space")

        one_hot_low = np.zeros(len(self.automaton_states), dtype=np.float32)
        one_hot_high = np.ones(len(self.automaton_states), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=np.concatenate(
                (env.observation_space.low.astype(np.float32), one_hot_low)
            ),
            high=np.concatenate(
                (env.observation_space.high.astype(np.float32), one_hot_high)
            ),
            dtype=np.float32,
        )

        self.episode_metrics = []
        self.current_q = None
        self.raw_observation = None
        self._initially_accepted = False
        self._episode = None

    def _abstract_position(self, observation):
        """Map a raw LunarLander observation to abstract grid coordinates."""
        x, y, _ = phi_mapping_sequential(
            observation,
            0,
            self.abstract_mdp.width,
            self.abstract_mdp.height,
        )
        return x, y

    def _augment_observation(self, observation, q):
        """Append a one-hot encoding of the active DFA state."""
        if q not in self.state_to_index:
            raise RuntimeError(f"DFA returned unknown state {q!r}")
        one_hot = np.zeros(len(self.automaton_states), dtype=np.float32)
        one_hot[self.state_to_index[q]] = 1.0
        return np.concatenate((observation, one_hot)).astype(np.float32)

    def _initial_q(self, observation):
        """Consume the initial observation before the first agent action."""
        x, y = self._abstract_position(observation)
        truth_assignment = self.abstract_mdp._get_truth_assignment(x, y)
        pre_trace_q = self.automaton.get_initial_q()
        return self.automaton.get_next_q(pre_trace_q, truth_assignment)

    def _new_episode_metrics(self, q):
        """Create the mutable counters for one episode."""
        state_visits = [0] * len(self.automaton_states)
        state_entries = [0] * len(self.automaton_states)
        state_visits[self.state_to_index[q]] = 1
        state_entries[self.state_to_index[q]] = 1
        return {
            "task_reward": 0.0,
            "shaping_reward": 0.0,
            "learning_reward": 0.0,
            "episode_length": 0,
            "success": False,
            "initial_acceptance": False,
            "abstract_changes": 0,
            "dfa_transitions": 0,
            "transition_counter": Counter(),
            "state_visits": state_visits,
            "state_entries": state_entries,
            "env_terminated": False,
            "env_truncated": False,
        }

    def reset(self, **kwargs):
        """Reset LunarLander and initialize the DFA from the first observation."""
        observation, info = self.env.reset(**kwargs)
        self.raw_observation = observation
        self.current_q = self._initial_q(observation)
        self._episode = self._new_episode_metrics(self.current_q)
        self._initially_accepted = self.automaton.is_goal_reached(self.current_q)
        return self._augment_observation(observation, self.current_q), info

    def _finish_episode(self, info):
        """Freeze episode metrics and expose a compact copy through ``info``."""
        completed = self._episode.copy()
        completed["transition_counter"] = Counter(
            self._episode["transition_counter"]
        )
        completed["state_visits"] = list(self._episode["state_visits"])
        completed["state_entries"] = list(self._episode["state_entries"])
        self.episode_metrics.append(completed)
        info["ltlf_episode"] = {
            key: value
            for key, value in completed.items()
            if key not in {"transition_counter", "state_visits", "state_entries"}
        }

    def step(self, action):
        """Advance LunarLander and replace its reward with the LTLf task reward."""
        if self.current_q is None or self._episode is None:
            raise RuntimeError("reset() must be called before step()")

        # Gymnasium cannot mark a reset observation as terminal.  If s0 already
        # satisfies the task, emit one synthetic terminal transition without
        # applying the proposed action to LunarLander.
        if self._initially_accepted:
            self._initially_accepted = False
            self._episode["task_reward"] = self.goal_reward
            self._episode["learning_reward"] = self.goal_reward
            self._episode["success"] = True
            self._episode["initial_acceptance"] = True
            info = {"environment_reward": 0.0, "is_success": True}
            self._finish_episode(info)
            return (
                self._augment_observation(self.raw_observation, self.current_q),
                self.goal_reward,
                True,
                False,
                info,
            )

        previous_observation = self.raw_observation
        previous_q = self.current_q
        next_observation, environment_reward, env_terminated, env_truncated, info = (
            self.env.step(action)
        )
        info = dict(info)
        info["environment_reward"] = float(environment_reward)

        x, y = self._abstract_position(previous_observation)
        next_x, next_y = self._abstract_position(next_observation)
        abstract_state = (x, y, previous_q)

        truth_assignment = self.abstract_mdp._get_truth_assignment(next_x, next_y)
        next_q = self.automaton.get_next_q(previous_q, truth_assignment)
        if next_q not in self.state_to_index:
            raise RuntimeError(f"DFA returned unknown state {next_q!r}")
        abstract_next_state = (next_x, next_y, next_q)

        self._episode["state_visits"][self.state_to_index[next_q]] += 1
        if abstract_state != abstract_next_state:
            self._episode["abstract_changes"] += 1

        if next_q != previous_q:
            self._episode["dfa_transitions"] += 1
            self._episode["state_entries"][self.state_to_index[next_q]] += 1
            self._episode["transition_counter"][(previous_q, next_q)] += 1

        success = self.automaton.is_goal_reached(next_q)
        task_reward = self.goal_reward if success else 0.0
        shaping_reward = 0.0
        if self.use_shaping and abstract_state != abstract_next_state:
            phi_state = self.abstract_mdp.v_star.get(abstract_state, 0.0)
            phi_next_state = self.abstract_mdp.v_star.get(abstract_next_state, 0.0)
            shaping_reward = self.shaping_scale * (
                self.abstract_mdp.gamma * phi_next_state - phi_state
            )
        learning_reward = task_reward + shaping_reward

        self._episode["task_reward"] += task_reward
        self._episode["shaping_reward"] += shaping_reward
        self._episode["learning_reward"] += learning_reward
        self._episode["episode_length"] += 1
        self._episode["success"] = success
        self._episode["env_terminated"] = bool(env_terminated)
        self._episode["env_truncated"] = bool(env_truncated)

        self.raw_observation = next_observation
        self.current_q = next_q
        terminated = bool(env_terminated or success)
        truncated = bool(env_truncated)
        info["is_success"] = success

        if terminated or truncated:
            self._finish_episode(info)

        return (
            self._augment_observation(next_observation, next_q),
            float(learning_reward),
            terminated,
            truncated,
            info,
        )
