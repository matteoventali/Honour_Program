"""Counterfactual shaping diagnostics written independently from training logs."""

import json
from pathlib import Path


class ShapingDiagnostics:
    """Measure alternative shaping formulas without changing learner rewards."""

    def __init__(
        self,
        filename,
        abstract_mdp,
        episodes,
        log_interval,
        seed,
        use_shaping,
        training_shaping_gamma,
        shaping_frequency,
        reward_scale,
        potential_mode,
        training_gamma=None,
    ):
        self.abstract_mdp = abstract_mdp
        self.abstract_gamma = float(abstract_mdp.gamma)
        self.gamma = float(
            self.abstract_gamma if training_gamma is None else training_gamma
        )
        self.episodes = int(episodes)
        self.log_interval = int(log_interval)
        self.shaping_frequency = shaping_frequency
        self.reward_scale = float(reward_scale)
        if self.reward_scale <= 0.0:
            raise ValueError("reward_scale must be greater than zero")
        self.optimal_successors_cache = {}
        self.potential_mode = potential_mode
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("w", encoding="utf-8")
        self._write(
            {
                "record_type": "metadata",
                "schema_version": 4,
                "seed": int(seed) if seed is not None else None,
                "episodes": self.episodes,
                "gamma": self.gamma,
                "ground_gamma": self.gamma,
                "abstract_gamma": self.abstract_gamma,
                "learner_uses_shaping": bool(use_shaping),
                "learner_uses_gamma": bool(training_shaping_gamma),
                "shaping_frequency": shaping_frequency,
                "terminal_potential_zero": True,
                "reward_scale": self.reward_scale,
                "training_rewards_normalized": self.reward_scale != 1.0,
                "potential_mode": potential_mode,
                "classic_formula": "gamma*Phi(next)-Phi(state)",
                "no_gamma_formula": "Phi(next)-Phi(state)",
                "detail_policy": (
                    "all transitions in episode 1, the final episode and every "
                    "log_interval episode; every q change in all other episodes"
                ),
            }
        )

    def _write(self, record):
        self.handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    def _is_sampled_episode(self, episode):
        return (
            episode == 0
            or episode + 1 == self.episodes
            or (episode + 1) % self.log_interval == 0
        )

    def _optimal_successors(self, state):
        if state not in self.optimal_successors_cache:
            candidates = []
            for action in self.abstract_mdp.actions:
                next_state, reward = self.abstract_mdp.get_transitions(state, action)
                inter_level_reward = (
                    self.abstract_mdp.get_inter_level_shaping_reward(state, next_state)
                    if hasattr(self.abstract_mdp, "get_inter_level_shaping_reward")
                    else 0.0
                )
                value = (
                    float(reward)
                    + float(inter_level_reward)
                    + self.gamma * float(self.abstract_mdp.v_star.get(next_state, 0.0))
                )
                candidates.append((next_state, value))
            best_value = max(value for _, value in candidates)
            tolerance = 1e-7 * max(1.0, abs(best_value))
            self.optimal_successors_cache[state] = {
                next_state
                for next_state, value in candidates
                if abs(value - best_value) <= tolerance
            }
        return self.optimal_successors_cache[state]

    @staticmethod
    def _signed_label(value, scale):
        tolerance = 1e-9 * max(1.0, scale)
        if value > tolerance:
            return "positive"
        if value < -tolerance:
            return "negative"
        return "zero"

    def start_episode(self, episode, initial_state, initial_raw_phi=None):
        if initial_raw_phi is None:
            initial_raw_phi = self.abstract_mdp.v_star.get(initial_state, 0.0)
        initial_raw_phi = float(initial_raw_phi)
        initial_phi = initial_raw_phi / self.reward_scale
        self.episode = int(episode)
        self.sampled_episode = self._is_sampled_episode(episode)
        self.initial_phi = initial_phi
        self.initial_raw_phi = initial_raw_phi
        self.final_phi = initial_phi
        self.steps = 0
        self.classic_sum = 0.0
        self.no_gamma_sum = 0.0
        self.discounted_classic_sum = 0.0
        self.same_abstract_steps = 0
        self.classic_same_abstract_sum = 0.0
        self.same_abstract_potential_trend_counts = {"increasing": 0, "equal": 0, "decreasing": 0}
        self.same_abstract_classic_sign_counts = {"positive": 0, "zero": 0, "negative": 0}
        self.abstract_changes = 0
        self.classic_abstract_change_sum = 0.0
        self.spatial_changes = 0
        self.classic_spatial_change_sum = 0.0
        self.q_changes = 0
        self.negative_q_changes = 0
        self.classic_q_change_sum = 0.0
        self.min_q_change_shaping = None
        self.truth_mismatch_steps = 0
        self.cell_only_truth_steps = 0
        self.real_only_truth_steps = 0
        self.truth_mismatch_by_proposition = {}
        self.current_dwell_steps = 0
        self.max_dwell_steps = 0
        self.potential_trend_counts = {"increasing": 0, "equal": 0, "decreasing": 0}
        self.classic_sign_counts = {"positive": 0, "zero": 0, "negative": 0}
        self.spatial_potential_trend_counts = {"increasing": 0, "equal": 0, "decreasing": 0}
        self.optimal_spatial_changes = 0
        self.optimal_spatial_classic_sum = 0.0
        self.optimal_spatial_sign_counts = {"positive": 0, "zero": 0, "negative": 0}
        self.nonoptimal_spatial_changes = 0
        self.nonoptimal_spatial_classic_sum = 0.0
        self.nonoptimal_spatial_sign_counts = {"positive": 0, "zero": 0, "negative": 0}
        self.terminal_spatial_changes = 0
        self.terminal_step_classic = 0.0
        self.terminated = False
        self.truncated = False

    def record_transition(
        self,
        step,
        state,
        next_state,
        raw_phi_state,
        raw_phi_next,
        real_truth,
        cell_truth,
        abstract_changed,
        spatial_changed,
        q_changed,
        applied_shaping,
        episode_done,
        bootstrap_terminal,
        env_terminated,
        env_truncated,
    ):
        raw_phi_state = float(raw_phi_state)
        raw_phi_next = float(raw_phi_next)
        phi_state = raw_phi_state / self.reward_scale
        phi_next = raw_phi_next / self.reward_scale
        effective_phi_next = 0.0 if bootstrap_terminal else phi_next
        classic = self.gamma * effective_phi_next - phi_state
        no_gamma = effective_phi_next - phi_state
        raw_potential_difference = phi_next - phi_state
        potential_sign = self._signed_label(
            raw_potential_difference,
            max(abs(phi_state), abs(phi_next)),
        )
        potential_trend = {
            "positive": "increasing",
            "zero": "equal",
            "negative": "decreasing",
        }[potential_sign]
        classic_sign = self._signed_label(classic, max(abs(phi_state), abs(phi_next)))
        abstract_optimal = (
            next_state in self._optimal_successors(state)
            if spatial_changed and not bootstrap_terminal
            else None
        )
        mismatch_props = sorted(
            name
            for name in set(real_truth) | set(cell_truth)
            if bool(real_truth.get(name, False)) != bool(cell_truth.get(name, False))
        )
        cell_only_props = sorted(
            name
            for name in set(real_truth) | set(cell_truth)
            if bool(cell_truth.get(name, False)) and not bool(real_truth.get(name, False))
        )
        real_only_props = sorted(
            name
            for name in set(real_truth) | set(cell_truth)
            if bool(real_truth.get(name, False)) and not bool(cell_truth.get(name, False))
        )

        self.steps += 1
        self.final_phi = effective_phi_next
        self.classic_sum += classic
        self.no_gamma_sum += no_gamma
        self.discounted_classic_sum += (self.gamma ** int(step)) * classic
        self.potential_trend_counts[potential_trend] += 1
        self.classic_sign_counts[classic_sign] += 1

        if abstract_changed:
            self.abstract_changes += 1
            self.classic_abstract_change_sum += classic
            self.current_dwell_steps = 0
        else:
            self.same_abstract_steps += 1
            self.classic_same_abstract_sum += classic
            self.same_abstract_potential_trend_counts[potential_trend] += 1
            self.same_abstract_classic_sign_counts[classic_sign] += 1
            self.current_dwell_steps += 1
            self.max_dwell_steps = max(self.max_dwell_steps, self.current_dwell_steps)
        if spatial_changed:
            self.spatial_changes += 1
            self.classic_spatial_change_sum += classic
            self.spatial_potential_trend_counts[potential_trend] += 1
            if bootstrap_terminal:
                self.terminal_spatial_changes += 1
            elif abstract_optimal:
                self.optimal_spatial_changes += 1
                self.optimal_spatial_classic_sum += classic
                self.optimal_spatial_sign_counts[classic_sign] += 1
            else:
                self.nonoptimal_spatial_changes += 1
                self.nonoptimal_spatial_classic_sum += classic
                self.nonoptimal_spatial_sign_counts[classic_sign] += 1
        if q_changed:
            self.q_changes += 1
            self.classic_q_change_sum += classic
            if classic < 0.0:
                self.negative_q_changes += 1
            self.min_q_change_shaping = (
                classic
                if self.min_q_change_shaping is None
                else min(self.min_q_change_shaping, classic)
            )
        if mismatch_props:
            self.truth_mismatch_steps += 1
            for name in mismatch_props:
                self.truth_mismatch_by_proposition[name] = (
                    self.truth_mismatch_by_proposition.get(name, 0) + 1
                )
        if cell_only_props:
            self.cell_only_truth_steps += 1
        if real_only_props:
            self.real_only_truth_steps += 1
        if episode_done:
            self.terminal_step_classic = classic
        self.terminated = self.terminated or bool(env_terminated)
        self.truncated = self.truncated or bool(env_truncated)

        if self.sampled_episode or q_changed:
            self._write(
                {
                    "record_type": "transition",
                    "episode": self.episode + 1,
                    "step": int(step) + 1,
                    "sampled_episode": self.sampled_episode,
                    "state": list(state),
                    "next_state": list(next_state),
                    "phi": phi_state,
                    "phi_next": phi_next,
                    "raw_phi": raw_phi_state,
                    "raw_phi_next": raw_phi_next,
                    "effective_phi_next": effective_phi_next,
                    "classic_shaping": classic,
                    "classic_sign": classic_sign,
                    "no_gamma_shaping": no_gamma,
                    "potential_trend": potential_trend,
                    "abstract_optimal_spatial_change": abstract_optimal,
                    "applied_shaping": float(applied_shaping),
                    "abstract_changed": bool(abstract_changed),
                    "spatial_changed": bool(spatial_changed),
                    "q_changed": bool(q_changed),
                    "real_truth": {key: bool(value) for key, value in real_truth.items()},
                    "cell_truth": {key: bool(value) for key, value in cell_truth.items()},
                    "truth_mismatch": mismatch_props,
                    "cell_only_truth": cell_only_props,
                    "real_only_truth": real_only_props,
                    "episode_done": bool(episode_done),
                    "bootstrap_terminal": bool(bootstrap_terminal),
                    "env_terminated": bool(env_terminated),
                    "env_truncated": bool(env_truncated),
                }
            )

    def finish_episode(self, succeeded, applied_shaping_sum):
        terminal_residual = (self.gamma ** self.steps) * self.final_phi
        telescoping_boundary = -self.initial_phi + terminal_residual
        self._write(
            {
                "record_type": "episode_summary",
                "episode": self.episode + 1,
                "sampled_episode": self.sampled_episode,
                "steps": self.steps,
                "succeeded": bool(succeeded),
                "env_terminated": self.terminated,
                "env_truncated": self.truncated,
                "initial_phi": self.initial_phi,
                "initial_raw_phi": self.initial_raw_phi,
                "final_phi": self.final_phi,
                "applied_shaping_sum": float(applied_shaping_sum),
                "classic_sum": self.classic_sum,
                "no_gamma_sum": self.no_gamma_sum,
                "discounted_classic_sum": self.discounted_classic_sum,
                "telescoping_boundary": telescoping_boundary,
                "telescoping_error": self.discounted_classic_sum - telescoping_boundary,
                "terminal_residual": terminal_residual,
                "zero_endpoint_boundary": -self.initial_phi,
                "terminal_step_classic": self.terminal_step_classic,
                "same_abstract_steps": self.same_abstract_steps,
                "classic_same_abstract_sum": self.classic_same_abstract_sum,
                "classic_same_abstract_mean": (
                    self.classic_same_abstract_sum / self.same_abstract_steps
                    if self.same_abstract_steps
                    else 0.0
                ),
                "same_abstract_potential_trend_counts": self.same_abstract_potential_trend_counts,
                "same_abstract_classic_sign_counts": self.same_abstract_classic_sign_counts,
                "abstract_changes": self.abstract_changes,
                "classic_abstract_change_sum": self.classic_abstract_change_sum,
                "spatial_changes": self.spatial_changes,
                "classic_spatial_change_sum": self.classic_spatial_change_sum,
                "q_changes": self.q_changes,
                "negative_q_changes": self.negative_q_changes,
                "classic_q_change_sum": self.classic_q_change_sum,
                "classic_q_change_mean": (
                    self.classic_q_change_sum / self.q_changes
                    if self.q_changes
                    else 0.0
                ),
                "min_q_change_shaping": self.min_q_change_shaping,
                "truth_mismatch_steps": self.truth_mismatch_steps,
                "truth_mismatch_fraction": (
                    self.truth_mismatch_steps / self.steps if self.steps else 0.0
                ),
                "cell_only_truth_steps": self.cell_only_truth_steps,
                "real_only_truth_steps": self.real_only_truth_steps,
                "truth_mismatch_by_proposition": self.truth_mismatch_by_proposition,
                "max_same_abstract_dwell": self.max_dwell_steps,
                "potential_trend_counts": self.potential_trend_counts,
                "classic_sign_counts": self.classic_sign_counts,
                "spatial_potential_trend_counts": self.spatial_potential_trend_counts,
                "optimal_spatial_changes": self.optimal_spatial_changes,
                "optimal_spatial_classic_sum": self.optimal_spatial_classic_sum,
                "optimal_spatial_classic_mean": (
                    self.optimal_spatial_classic_sum / self.optimal_spatial_changes
                    if self.optimal_spatial_changes
                    else 0.0
                ),
                "optimal_spatial_sign_counts": self.optimal_spatial_sign_counts,
                "nonoptimal_spatial_changes": self.nonoptimal_spatial_changes,
                "nonoptimal_spatial_classic_sum": self.nonoptimal_spatial_classic_sum,
                "nonoptimal_spatial_classic_mean": (
                    self.nonoptimal_spatial_classic_sum / self.nonoptimal_spatial_changes
                    if self.nonoptimal_spatial_changes
                    else 0.0
                ),
                "nonoptimal_spatial_sign_counts": self.nonoptimal_spatial_sign_counts,
                "terminal_spatial_changes": self.terminal_spatial_changes,
            }
        )
        self.handle.flush()

    def close(self):
        self.handle.close()
