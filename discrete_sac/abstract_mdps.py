import re
from collections import defaultdict
import numpy as np

# Import the LTLf parser provided by ltlf2dfa.
from ltlf2dfa.parser.ltlf import LTLfParser
from graphviz import Source

class LTLfAutomaton:
    """
    Wrap ltlf2dfa and expose its DFA as a graph that can be traversed by the MDP.
    """
    def __init__(self, formula_str):
        self.formula_str = formula_str
        
        # Parse the formula and generate its DFA in DOT format.
        parser = LTLfParser()
        parsed_formula = parser(formula_str)
        dot_string = parsed_formula.to_dfa()
        self.dot_string = parsed_formula.to_dfa()
        
        # Initialize the automaton data structures.
        self.states = set()
        self.accepting_states = set()
        self.transitions = {}  # {source_state: [(Boolean_guard, destination_state), ...]}
        self.initial_state = None
        
        # Extract states and transitions from the DOT representation.
        self._parse_dot(dot_string)
        
        # Keep a stable state order for the MDP and one-hot encodings.
        self.states = sorted(list(self.states))
        self.num_phases = len(self.states)

    def _parse_dot(self, dot_string):
        """
        Parse the DOT output and extract states, accepting states, the initial
        state, and guarded transitions.
        """
        # Extract accepting states, e.g. node [shape = doublecircle]; 2 3;.
        match_acc = re.search(r'node\s*\[shape\s*=\s*doublecircle\]\s*;\s*(.*?);', dot_string)
        if match_acc:
            acc_str = match_acc.group(1).replace(',', ' ')
            self.accepting_states = set(int(s) for s in acc_str.split() if s.strip().isdigit())
            
        # Extract guarded transitions, e.g. 1 -> 2 [label="wp1 & ~wp2"].
        trans_matches = re.findall(r'(\d+)\s*->\s*(\d+)\s*\[label\s*=\s*"(.*?)"\]', dot_string)
        for src_str, dst_str, guard in trans_matches:
            src = int(src_str)
            dst = int(dst_str)
            self.states.add(src)
            self.states.add(dst)
            
            if src not in self.transitions:
                self.transitions[src] = []
            self.transitions[src].append((guard, dst))
            
        # Extract the initial state from the unlabeled edge leaving the invisible node.
        # Example: 0 [style=invis]; 0 -> 1;.
        init_match = re.search(r'(\d+)\s*->\s*(\d+)\s*;', dot_string)
        if init_match:
            self.initial_state = int(init_match.group(2))
        else:
            self.initial_state = min(self.states) if self.states else 0

    def get_initial_q(self):
        """Return the identifier of the DFA pre-trace state."""
        return self.initial_state

    def is_goal_reached(self, current_q):
        """Return whether the current DFA state is accepting."""
        return current_q in self.accepting_states

    def get_next_q(self, current_q, truth_assignment):
        """
        Evaluate outgoing transition guards and return the next DFA state.
        """
        if current_q not in self.transitions:
            return current_q
            
        for guard, next_q in self.transitions[current_q]:
            if self._eval_guard(guard, truth_assignment):
                return next_q
                
        return current_q

    def _eval_guard(self, guard, truth_assignment):
        """
        Convert a DOT guard such as "wp1 & ~wp2" to Python syntax and evaluate
        it against the current truth assignment.
        """
        guard = guard.strip()
        
        # Handle numeric and textual Boolean constants.
        if guard.lower() in ["1", "true"]: return True
        if guard.lower() in ["0", "false"]: return False
        
        # Convert the standard Boolean operators to Python syntax.
        expr = guard.replace('&', ' and ').replace('|', ' or ').replace('~', ' not ').replace('!', ' not ')
        
        try:
            # Disable built-ins while evaluating the Boolean expression.
            return eval(expr, {"__builtins__": {}}, truth_assignment)
        except Exception as e:
            print(f"[LTLfAutomaton error] Could not evaluate transition guard '{guard}': {e}")
            return False

    def render_graph(self, filename="ltlf_automaton", directory="img"):
        """Render the DFA and save it as a PNG image."""
        try:
            # ltlf2dfa emits a left-to-right graph.  With complex formulae the
            # transition guards become wide, leaving the resulting PNG only a
            # few pixels high.  A top-to-bottom layout gives labels enough room
            # and keeps the automaton readable independently of formula length.
            render_dot = re.sub(
                r"rankdir\s*=\s*LR\s*;",
                "rankdir = TB;",
                self.dot_string,
                count=1,
            )
            render_dot = re.sub(
                r"(digraph[^{]*\{)",
                (
                    r"\1\n"
                    r'graph [pad="0.35", nodesep="0.55", ranksep="0.75"];' "\n"
                    r'node [width="0.55", height="0.55"];' "\n"
                    r'edge [fontsize="10"];'
                ),
                render_dot,
                count=1,
            )
            src = Source(render_dot)
            src.render(filename=filename, directory=directory, format='png', cleanup=True)
            print(f"Automaton graph saved to: {directory}/{filename}.png")
        except Exception as e:
            print(f"[Graphviz error] Could not render the automaton graph: {e}")


class LTLfWaypointMDP:
    """
    Abstract MDP guided by an LTLf automaton.
    Each abstract state is (x, y, q), where q is the DFA state identifier.
    """
    def __init__(self, waypoints_dict, ltlf_automaton, width=12, height=12, gamma=0.99, goal_reward=10000):
        self.width = width
        self.height = height
        self.gamma = gamma
        self.actions = [0, 1, 2, 3, 4, 5, 6, 7] # Include diagonal movements.
        
        self.waypoints_dict = waypoints_dict
        self.automaton = ltlf_automaton
        self.num_phases = self.automaton.num_phases
        
        # Generate every combination of grid position and DFA state.
        self.states = [(x, y, q) for x in range(width) for y in range(height) for q in self.automaton.states]
        
        self.goal_reward = goal_reward
        self.v_star = defaultdict(float)
        
    def _get_truth_assignment(self, x, y):
        """
        Map the current grid coordinates to a Boolean proposition assignment.
        """
        truth_assignment = {}
        for prop_name, (wp_x, wp_y) in self.waypoints_dict.items():
            truth_assignment[prop_name] = (x == wp_x and y == wp_y)
        return truth_assignment

    def get_transitions(self, state, action):
        x, y, q = state
        reward = 0
        
        # Apply the abstract physical movement.
        next_y = y
        if action in [0, 4, 5]:    next_y = min(y + 1, self.height - 1)
        elif action in [1, 6, 7]:  next_y = max(y - 1, 0)
            
        next_x = x
        if action in [2, 4, 6]:    next_x = max(x - 1, 0)
        elif action in [3, 5, 7]:  next_x = min(x + 1, self.width - 1)
        
        # Evaluate propositions at the arrival coordinates.
        truth_assignment = self._get_truth_assignment(next_x, next_y)
        
        # Advance the automaton using the arrival-state valuation.
        next_q = self.automaton.get_next_q(q, truth_assignment)

        next_state = (next_x, next_y, next_q)
        return next_state, reward

    def print_policy(self):
        arrows = {
            0: "↑",
            1: "↓",
            2: "←",
            3: "→",
            4: "↖",
            5: "↗",
            6: "↙",
            7: "↘"
        }

        for q in self.automaton.states:
            print(f"\n===== POLICY - DFA STATE q={q} =====")

            for y in reversed(range(self.height)):
                row = []

                for x in range(self.width):
                    state = (x, y, q)

                    if self.automaton.is_goal_reached(q):
                        row.append(" G ")
                        continue

                    best_action = None
                    best_value = -float("inf")

                    for a in self.actions:
                        next_state, reward = self.get_transitions(state, a)
                        value = reward + self.gamma * self.v_star[next_state]

                        if value > best_value:
                            best_value = value
                            best_action = a

                    row.append(f" {arrows[best_action]} ")

                print("".join(row))
    
    def value_iteration(self, theta=0.001):
        print(f"Value Iteration...")
        
        for s in self.states:
            if self.automaton.is_goal_reached(s[2]):
                self.v_star[s] = self.goal_reward
        
        while True:
            delta = 0
            new_v = self.v_star.copy()
            for s in self.states:
                if not self.automaton.is_goal_reached(s[2]):
                    v_actions = [self.get_transitions(s, a)[1] + self.gamma * self.v_star[self.get_transitions(s, a)[0]] for a in self.actions]
                    best_v = max(v_actions)
                    delta = max(delta, abs(best_v - self.v_star[s]))
                    new_v[s] = best_v
            self.v_star = new_v
            if delta < theta: break

        #self.print_policy()
        
