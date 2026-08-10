"""Abstract grid MDP driven by a manually defined automaton."""

from collections import defaultdict


class ManualWaypointMDP:
    """Grid abstraction whose state is ``(x, y, q)``."""

    def __init__(
        self,
        waypoints_dict,
        automaton,
        width=12,
        height=12,
        gamma=0.99,
        goal_reward=10000,
    ):
        self.width = width
        self.height = height
        self.gamma = gamma
        self.actions = [0, 1, 2, 3, 4, 5, 6, 7]
        self.waypoints_dict = waypoints_dict
        self.automaton = automaton
        self.num_phases = self.automaton.num_phases
        self.states = [
            (x, y, q)
            for x in range(width)
            for y in range(height)
            for q in self.automaton.active_states
        ]
        self.goal_reward = goal_reward
        self.v_star = defaultdict(float)

    def _get_truth_assignment(self, x, y):
        """Map grid coordinates to waypoint proposition values."""
        return {
            proposition: (x == waypoint_x and y == waypoint_y)
            for proposition, (waypoint_x, waypoint_y) in self.waypoints_dict.items()
        }

    def get_transitions(self, state, action):
        """Apply one abstract movement and one automaton transition."""
        x, y, q = state

        next_y = y
        if action in [0, 4, 5]:
            next_y = min(y + 1, self.height - 1)
        elif action in [1, 6, 7]:
            next_y = max(y - 1, 0)

        next_x = x
        if action in [2, 4, 6]:
            next_x = max(x - 1, 0)
        elif action in [3, 5, 7]:
            next_x = min(x + 1, self.width - 1)

        truth_assignment = self._get_truth_assignment(next_x, next_y)
        automaton_step = self.automaton.advance(q, truth_assignment)
        reward = float(self.goal_reward) if automaton_step.completed_cycle else 0.0
        return (next_x, next_y, automaton_step.next_state), reward

    def print_policy(self):
        arrows = {
            0: "↑",
            1: "↓",
            2: "←",
            3: "→",
            4: "↖",
            5: "↗",
            6: "↙",
            7: "↘",
        }

        for q in self.automaton.active_states:
            print(f"\n===== POLICY - AUTOMATON STATE q={q} =====")
            for y in reversed(range(self.height)):
                row = []
                for x in range(self.width):
                    state = (x, y, q)
                    best_action = max(
                        self.actions,
                        key=lambda action: (
                            self.get_transitions(state, action)[1]
                            + self.gamma
                            * self.v_star[self.get_transitions(state, action)[0]]
                        ),
                    )
                    row.append(f" {arrows[best_action]} ")
                print("".join(row))

    def value_iteration(self, theta=0.001):
        """Compute continuing-task values, including every future goal cycle."""
        print("Value Iteration...")

        while True:
            delta = 0.0
            new_v = self.v_star.copy()
            for state in self.states:
                action_values = []
                for action in self.actions:
                    next_state, reward = self.get_transitions(state, action)
                    action_values.append(
                        reward + self.gamma * self.v_star[next_state]
                    )
                best_value = max(action_values)
                delta = max(delta, abs(best_value - self.v_star[state]))
                new_v[state] = best_value
            self.v_star = new_v
            if delta < theta:
                break

        self.print_policy()
