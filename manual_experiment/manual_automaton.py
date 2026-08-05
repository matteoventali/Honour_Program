"""Manual finite-state automata used by the experiment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AutomatonStep:
    """Result of one valuation, after exhausting automatic transitions."""

    next_state: str
    completed_cycle: bool = False
    reached_waypoint: str | None = None


class CyclicWaypointsAutomaton:
    """Automaton that repeatedly visits an ordered sequence of waypoints.

    Every stable state ``qi`` waits for the waypoint at position ``i`` in
    ``waypoint_cycle``. Reaching the last waypoint enters a transient accepting
    state and immediately resets to ``q1`` without consuming another
    environment step.
    """

    def __init__(self, waypoint_cycle: Sequence[str]):
        if isinstance(waypoint_cycle, (str, bytes)):
            raise TypeError("waypoint_cycle must be a sequence of proposition names")

        cycle = tuple(waypoint_cycle)
        if not cycle:
            raise ValueError("waypoint_cycle must contain at least one waypoint")
        if any(not isinstance(name, str) or not name for name in cycle):
            raise ValueError("Waypoint proposition names must be non-empty strings")
        if len(set(cycle)) != len(cycle):
            raise ValueError("waypoint_cycle cannot contain duplicate propositions")

        self.waypoint_cycle = cycle
        self.active_states = tuple(f"q{index}" for index in range(1, len(cycle) + 1))
        self.accepting_state = f"q{len(cycle) + 1}"
        self.states = (*self.active_states, self.accepting_state)
        self.initial_state = self.active_states[0]
        self.accepting_states = frozenset({self.accepting_state})
        self.num_phases = len(self.active_states)
        self._state_to_index = {
            state: index for index, state in enumerate(self.active_states)
        }

    @property
    def required_propositions(self) -> frozenset[str]:
        """Return the waypoint propositions consumed by this automaton."""
        return frozenset(self.waypoint_cycle)

    def get_initial_q(self) -> str:
        """Return the state active at the beginning of every episode."""
        return self.initial_state

    def is_accepting(self, state: str) -> bool:
        """Return whether ``state`` is the transient accepting state."""
        self._validate_state(state)
        return state in self.accepting_states

    def expected_waypoint(self, state: str) -> str:
        """Return the waypoint proposition awaited in a stable state."""
        self._validate_state(state)
        if state == self.accepting_state:
            raise ValueError("The transient accepting state does not await a waypoint")
        return self.waypoint_cycle[self._state_to_index[state]]

    def advance(self, current_q: str, truth_assignment: Mapping[str, bool]) -> AutomatonStep:
        """Consume one valuation and exhaust the accepting epsilon transition."""
        self._validate_state(current_q)

        # Also close the accepting state when explicitly supplied by diagnostics
        # or external code.
        if current_q == self.accepting_state:
            return AutomatonStep(self.initial_state)

        phase_index = self._state_to_index[current_q]
        expected = self.waypoint_cycle[phase_index]
        if not bool(truth_assignment.get(expected, False)):
            return AutomatonStep(current_q)

        if phase_index == len(self.waypoint_cycle) - 1:
            return AutomatonStep(
                self.initial_state,
                completed_cycle=True,
                reached_waypoint=expected,
            )

        return AutomatonStep(
            self.active_states[phase_index + 1],
            reached_waypoint=expected,
        )

    def get_next_q(self, current_q: str, truth_assignment: Mapping[str, bool]) -> str:
        """Return the stable state reached after applying epsilon closure."""
        return self.advance(current_q, truth_assignment).next_state

    def validate_waypoints(self, waypoints: Mapping[str, tuple[int, int]], width: int, height: int) -> None:
        """Validate the propositions and coordinates required by the automaton."""
        if width <= 0 or height <= 0:
            raise ValueError("Grid dimensions must be positive")

        missing = sorted(self.required_propositions - set(waypoints))
        if missing:
            raise ValueError(f"Missing required cycle waypoints: {missing}")

        for name, coordinates in waypoints.items():
            if not isinstance(coordinates, (tuple, list)) or len(coordinates) != 2:
                raise ValueError(
                    f"Waypoint {name!r} must contain exactly two coordinates"
                )
            x, y = coordinates
            if not isinstance(x, int) or not isinstance(y, int):
                raise ValueError(f"Waypoint {name!r} coordinates must be integers")
            if not 0 <= x < width or not 0 <= y < height:
                raise ValueError(
                    f"Waypoint {name!r} at ({x}, {y}) is outside "
                    f"the {width}x{height} grid"
                )

    def describe_cycle(self) -> str:
        """Return a compact human-readable representation of the automaton."""
        transitions = [
            f"{state} --{waypoint}{'/reward' if index + 1 == self.num_phases else ''}--> "
            f"{self.active_states[index + 1] if index + 1 < self.num_phases else self.accepting_state}"
            for index, (state, waypoint) in enumerate(
                zip(self.active_states, self.waypoint_cycle)
            )
        ]
        transitions.append(f"{self.accepting_state} --epsilon--> {self.initial_state}")
        return " | ".join(transitions)

    def render_graph(self, filename: str = "cyclic_waypoints_automaton", directory: str | Path = "img") -> None:
        """Render a compact diagram when the optional Graphviz package is present."""
        try:
            from graphviz import Source

            lines = [
                "digraph cyclic_waypoints {",
                "    rankdir=LR;",
                "    node [shape=circle];",
                "    start [shape=point];",
                f"    {self.accepting_state} [shape=doublecircle];",
                f"    start -> {self.initial_state};",
            ]
            for index, (state, waypoint) in enumerate(
                zip(self.active_states, self.waypoint_cycle)
            ):
                destination = (
                    self.active_states[index + 1]
                    if index + 1 < self.num_phases
                    else self.accepting_state
                )
                escaped_waypoint = waypoint.replace("\\", "\\\\").replace('"', '\\"')
                reward_suffix = " / reward" if destination == self.accepting_state else ""
                lines.extend(
                    [
                        f'    {state} -> {destination} [label="{escaped_waypoint}{reward_suffix}"];',
                        f'    {state} -> {state} [label="not {escaped_waypoint}"];',
                    ]
                )
            lines.extend(
                [
                    f'    {self.accepting_state} -> {self.initial_state} [label="epsilon"];',
                    "}",
                ]
            )
            Source("\n".join(lines)).render(
                filename=filename,
                directory=str(directory),
                format="png",
                cleanup=True,
            )
            print(f"Automaton graph saved to: {directory}/{filename}.png")
        except Exception as error:
            print(f"[Graphviz error] Could not render the automaton graph: {error}")

    def _validate_state(self, state: str) -> None:
        if state not in self.states:
            raise ValueError(f"Unknown automaton state {state!r}")


class AlternatingGoalsAutomaton(CyclicWaypointsAutomaton):
    """Backward-compatible two-waypoint specialization."""

    WAITING_FOR_G1 = "q1"
    WAITING_FOR_G2 = "q2"
    ACCEPTING = "q3"

    def __init__(self, first_goal: str = "g1", second_goal: str = "g2"):
        super().__init__((first_goal, second_goal))
        self.first_goal = first_goal
        self.second_goal = second_goal

    def render_graph(self, filename: str = "alternating_goals_automaton", directory: str | Path = "img") -> None:
        """Render using the historical default filename."""
        super().render_graph(filename=filename, directory=directory)
