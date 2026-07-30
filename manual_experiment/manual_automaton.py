"""Manual finite-state automata used by the experiment."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AutomatonStep:
    """Result of one valuation, after exhausting automatic transitions."""

    next_state: str
    completed_cycle: bool = False


class AlternatingGoalsAutomaton:
    """Three-state automaton that repeatedly enforces ``g1`` followed by ``g2``.

    ``q1`` waits for ``g1``. ``q2`` waits for ``g2`` after ``g1`` has been
    observed. Entering ``q3`` completes one cycle and is accepting. An epsilon
    transition then resets immediately to ``q1`` without consuming another
    environment step.
    """

    WAITING_FOR_G1 = "q1"
    WAITING_FOR_G2 = "q2"
    ACCEPTING = "q3"

    def __init__(self, first_goal: str = "g1", second_goal: str = "g2"):
        if not first_goal or not second_goal:
            raise ValueError("Goal proposition names cannot be empty")
        if first_goal == second_goal:
            raise ValueError("The first and second goal propositions must be distinct")

        self.first_goal = first_goal
        self.second_goal = second_goal
        self.states = (
            self.WAITING_FOR_G1,
            self.WAITING_FOR_G2,
            self.ACCEPTING,
        )
        # Epsilon closure makes q3 transient: it is never exposed to the agent.
        self.active_states = (
            self.WAITING_FOR_G1,
            self.WAITING_FOR_G2,
        )
        self.initial_state = self.WAITING_FOR_G1
        self.accepting_states = frozenset({self.ACCEPTING})
        self.num_phases = len(self.active_states)

    @property
    def required_propositions(self) -> frozenset[str]:
        """Return the waypoint propositions consumed by this automaton."""
        return frozenset({self.first_goal, self.second_goal})

    def get_initial_q(self) -> str:
        """Return the state active at the beginning of every episode."""
        return self.initial_state

    def is_accepting(self, state: str) -> bool:
        """Return whether ``state`` is the accepting state."""
        self._validate_state(state)
        return state in self.accepting_states

    def advance(
        self, current_q: str, truth_assignment: Mapping[str, bool]
    ) -> AutomatonStep:
        """Consume one valuation and exhaust the accepting epsilon transition."""
        self._validate_state(current_q)

        if current_q == self.WAITING_FOR_G1:
            if bool(truth_assignment.get(self.first_goal, False)):
                return AutomatonStep(self.WAITING_FOR_G2)
            return AutomatonStep(self.WAITING_FOR_G1)

        if current_q == self.WAITING_FOR_G2:
            if bool(truth_assignment.get(self.second_goal, False)):
                # q2 -> q3 produces the event and q3 --epsilon--> q1 is
                # completed atomically in the same environment transition.
                return AutomatonStep(
                    self.WAITING_FOR_G1,
                    completed_cycle=True,
                )
            return AutomatonStep(self.WAITING_FOR_G2)

        # Also close q3 when explicitly supplied by diagnostics or external code.
        return AutomatonStep(self.WAITING_FOR_G1)

    def get_next_q(
        self, current_q: str, truth_assignment: Mapping[str, bool]
    ) -> str:
        """Return the stable state reached after applying epsilon closure."""
        return self.advance(current_q, truth_assignment).next_state

    def validate_waypoints(
        self,
        waypoints: Mapping[str, tuple[int, int]],
        width: int,
        height: int,
    ) -> None:
        """Validate the propositions and coordinates required by the automaton."""
        if width <= 0 or height <= 0:
            raise ValueError("Grid dimensions must be positive")

        missing = sorted(self.required_propositions - set(waypoints))
        if missing:
            raise ValueError(f"Missing required goal waypoints: {missing}")

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

    def render_graph(
        self,
        filename: str = "alternating_goals_automaton",
        directory: str | Path = "img",
    ) -> None:
        """Render a compact diagram when the optional Graphviz package is present."""
        try:
            from graphviz import Source

            dot = f"""
digraph alternating_goals {{
    rankdir=LR;
    node [shape=circle];
    start [shape=point];
    {self.ACCEPTING} [shape=doublecircle];
    start -> {self.WAITING_FOR_G1};
    {self.WAITING_FOR_G1} -> {self.WAITING_FOR_G2} [label="{self.first_goal}"];
    {self.WAITING_FOR_G1} -> {self.WAITING_FOR_G1} [label="not {self.first_goal}"];
    {self.WAITING_FOR_G2} -> {self.ACCEPTING} [label="{self.second_goal} / reward"];
    {self.WAITING_FOR_G2} -> {self.WAITING_FOR_G2} [label="not {self.second_goal}"];
    {self.ACCEPTING} -> {self.WAITING_FOR_G1} [label="epsilon"];
}}
"""
            Source(dot).render(
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
