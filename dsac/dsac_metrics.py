"""Episode metrics collected by the LTLf environment wrapper."""

from collections import defaultdict

import numpy as np


# =============================================================================
# Mutable training state
# =============================================================================

class TrainingMetrics:
    """Own all mutable training metrics without relying on module globals."""

    def __init__(self, expected_episodes=None, log_interval=100):
        self.expected_episodes = expected_episodes
        self.log_interval = log_interval
        self.enabled = False
        self.clear()

    def clear(self):
        """Discard all recorded episodes and DFA transitions."""
        self.task_rewards = []
        self.total_rewards = []
        self.episode_lengths = []
        self.episode_end_reasons = []
        self.dfa_transition_counts = defaultdict(int)

    def record_transition(self, source, destination):
        """Count a DFA transition when metric collection is enabled."""
        if self.enabled:
            self.dfa_transition_counts[(source, destination)] += 1

    def record_episode(self, task_reward, total_reward, length, end_reason, use_shaping):
        """Append one completed episode and print periodic diagnostics."""
        if not self.enabled:
            return
        self.task_rewards.append(task_reward)
        self.total_rewards.append(total_reward)
        self.episode_lengths.append(length)
        self.episode_end_reasons.append(end_reason)
        if len(self.task_rewards) % self.log_interval == 0:
            self._print_progress(use_shaping)

    def _print_progress(self, use_shaping):
        """Print recent and cumulative learning diagnostics."""
        window = min(self.log_interval, len(self.task_rewards))
        recent_rewards = np.asarray(self.task_rewards[-window:])
        recent_reasons = self.episode_end_reasons[-window:]
        successes = int(np.count_nonzero(np.asarray(self.task_rewards) > 0))
        transitions = ", ".join(
            f"{source}->{destination}: {count}"
            for (source, destination), count in sorted(self.dfa_transition_counts.items())
        ) or "none"
        mode = "SHAPING" if use_shaping else "BASELINE"
        total = self.expected_episodes if self.expected_episodes is not None else "?"
        training_reward_line = (
            f"Average training reward      : {np.mean(self.total_rewards[-window:]):.6f}\n"
            if use_shaping
            else ""
        )
        print(
            f"\n[{mode} | DSAC] Episode {len(self.task_rewards)}/{total}\n"
            f"Average task reward          : {np.mean(recent_rewards):.6f}\n"
            f"Cumulative successes         : {successes}/{len(self.task_rewards)} "
            f"({successes / len(self.task_rewards):.2%})\n"
            f"Success rate (last {window}) : {np.mean(recent_rewards > 0):.2%}\n"
            f"Endings (last {window})      : "
            f"environment={recent_reasons.count('environment_terminated')}, "
            f"truncated={recent_reasons.count('truncated')}, "
            f"success={recent_reasons.count('success')}\n"
            f"Average episode length       : {np.mean(self.episode_lengths[-window:]):.1f}\n"
            f"{training_reward_line}"
            f"DFA transitions (cumulative) : {transitions}"
        )

    def validate_episode_count(self, expected_episodes):
        """Ensure that every metric contains exactly one value per episode."""
        lengths = {
            "task_rewards": len(self.task_rewards),
            "total_rewards": len(self.total_rewards),
            "episode_lengths": len(self.episode_lengths),
            "episode_end_reasons": len(self.episode_end_reasons),
        }
        if any(length != expected_episodes for length in lengths.values()):
            raise RuntimeError(
                f"Expected metrics for {expected_episodes} episodes, recorded {lengths}."
            )

    def as_dict(self):
        """Return defensive copies suitable for serialization."""
        return {
            "task_rewards": self.task_rewards.copy(),
            "total_rewards": self.total_rewards.copy(),
            "episode_lengths": self.episode_lengths.copy(),
            "episode_end_reasons": self.episode_end_reasons.copy(),
            "dfa_transition_counts": dict(self.dfa_transition_counts),
        }
