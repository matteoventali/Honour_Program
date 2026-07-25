"""DFA validation shared with the LunarLander trainers."""

import itertools
import re
from collections import deque


LTLF_OPERATORS = {"F", "G", "M", "R", "U", "W", "X", "false", "true"}


def _extract_formula_propositions(formula):
    tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", formula))
    return sorted(token for token in tokens if token not in LTLF_OPERATORS)


def _generate_truth_assignments(propositions):
    for values in itertools.product((False, True), repeat=len(propositions)):
        yield dict(zip(propositions, values))


def _matching_transitions(automaton, state, truth_assignment):
    return [
        (guard, destination)
        for guard, destination in automaton.transitions.get(state, [])
        if automaton._eval_guard(guard, truth_assignment)
    ]


class AutomatonValidationReport:
    def __init__(self, formula, propositions):
        self.formula = formula
        self.propositions = propositions
        self.errors = []
        self.warnings = []
        self.statistics = {}

    @property
    def is_valid(self):
        return not self.errors

    def add_error(self, message):
        self.errors.append(message)

    def add_warning(self, message):
        self.warnings.append(message)

    def format(self):
        status = "VALID" if self.is_valid else "INVALID"
        lines = [
            "=== AUTOMATON VALIDATION ===",
            f"Status: {status}",
            f"Formula propositions: {self.propositions}",
            f"Statistics: {self.statistics}",
        ]
        if self.errors:
            lines.append("Errors:")
            lines.extend(f"- {message}" for message in self.errors)
        if self.warnings:
            lines.append("Warnings:")
            lines.extend(f"- {message}" for message in self.warnings)
        return "\n".join(lines)

    def raise_if_invalid(self):
        if not self.is_valid:
            raise ValueError(self.format())


def validate_automaton(
    automaton,
    waypoints_dict,
    width=None,
    height=None,
    max_propositions=12,
    raise_on_error=True,
):
    """Validate DFA structure, guards, reachability, propositions and coordinates."""
    propositions = _extract_formula_propositions(automaton.formula_str)
    report = AutomatonValidationReport(automaton.formula_str, propositions)
    states = set(automaton.states)
    waypoint_propositions = set(waypoints_dict)

    missing_waypoints = sorted(set(propositions) - waypoint_propositions)
    unused_waypoints = sorted(waypoint_propositions - set(propositions))
    if missing_waypoints:
        report.add_error(f"Formula propositions without coordinates: {missing_waypoints}")
    if unused_waypoints:
        report.add_warning(f"Waypoint propositions not used by the formula: {unused_waypoints}")
    if len(propositions) > max_propositions:
        report.add_error(
            f"The formula has {len(propositions)} propositions; "
            f"exhaustive validation is limited to {max_propositions}"
        )

    for proposition, coordinates in waypoints_dict.items():
        if not isinstance(coordinates, (tuple, list)) or len(coordinates) != 2:
            report.add_error(f"Waypoint {proposition!r} must contain exactly two coordinates")
            continue
        x, y = coordinates
        if not isinstance(x, int) or not isinstance(y, int):
            report.add_error(f"Waypoint {proposition!r} coordinates must be integers")
        if width is not None and not 0 <= x < width:
            report.add_error(f"Waypoint {proposition!r} has x={x}, outside [0, {width - 1}]")
        if height is not None and not 0 <= y < height:
            report.add_error(f"Waypoint {proposition!r} has y={y}, outside [0, {height - 1}]")

    if not states:
        report.add_error("The DFA contains no states")
    if automaton.get_initial_q() not in states:
        report.add_error(
            f"Initial state {automaton.get_initial_q()!r} is not part of the DFA"
        )
    unknown_accepting = sorted(set(automaton.accepting_states) - states)
    if unknown_accepting:
        report.add_error(f"Unknown accepting states: {unknown_accepting}")
    if not automaton.accepting_states:
        report.add_error("The DFA has no accepting states")
    for source, transitions in automaton.transitions.items():
        if source not in states:
            report.add_error(f"Transition source {source!r} is not part of the DFA")
        for guard, destination in transitions:
            if destination not in states:
                report.add_error(
                    f"Transition {source!r} --[{guard}]--> {destination!r} "
                    "targets an unknown state"
                )

    truth_assignments = (
        list(_generate_truth_assignments(propositions))
        if len(propositions) <= max_propositions
        else []
    )
    reachable_states = (
        {automaton.get_initial_q()} if automaton.get_initial_q() in states else set()
    )
    frontier = deque(reachable_states)
    checked_pairs = 0
    ambiguous_pairs = 0
    incomplete_pairs = 0

    for state in states:
        for truth_assignment in truth_assignments:
            matches = _matching_transitions(automaton, state, truth_assignment)
            checked_pairs += 1
            if not matches:
                incomplete_pairs += 1
                report.add_error(
                    f"No transition from state {state!r} for valuation {truth_assignment}"
                )
            elif len(matches) > 1:
                ambiguous_pairs += 1
                report.add_error(
                    f"Ambiguous transitions from state {state!r} for valuation "
                    f"{truth_assignment}: {[guard for guard, _ in matches]}"
                )
            else:
                expected_destination = matches[0][1]
                actual_destination = automaton.get_next_q(state, truth_assignment)
                if actual_destination != expected_destination:
                    report.add_error(
                        f"get_next_q returned {actual_destination!r}, expected "
                        f"{expected_destination!r} from state {state!r}"
                    )

    while frontier and truth_assignments:
        state = frontier.popleft()
        for truth_assignment in truth_assignments:
            matches = _matching_transitions(automaton, state, truth_assignment)
            if len(matches) == 1 and matches[0][1] not in reachable_states:
                reachable_states.add(matches[0][1])
                frontier.append(matches[0][1])

    unreachable_states = sorted(states - reachable_states)
    unreachable_accepting = sorted(set(automaton.accepting_states) - reachable_states)
    if unreachable_states:
        report.add_warning(f"Unreachable DFA states: {unreachable_states}")
    if unreachable_accepting:
        report.add_error(f"No accepting path reaches states: {unreachable_accepting}")

    report.statistics = {
        "states": len(states),
        "accepting_states": len(automaton.accepting_states),
        "transitions": sum(len(transitions) for transitions in automaton.transitions.values()),
        "valuations": len(truth_assignments),
        "state_valuation_pairs": checked_pairs,
        "ambiguous_pairs": ambiguous_pairs,
        "incomplete_pairs": incomplete_pairs,
        "reachable_states": len(reachable_states),
    }
    if raise_on_error:
        report.raise_if_invalid()
    return report
