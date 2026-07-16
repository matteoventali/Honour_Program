from collections import defaultdict
import numpy as np

class NPhaseWaypointMDP:
    """
    MDP for sequential tasks with N dynamic phases.
    The abstract state is 3D: (x, y, q)
    q ranges from 0 to len(waypoints) - 1.
    The final element in the waypoints list is considered the Final Goal.
    """
    def __init__(self, waypoints, width=12, height=12, gamma=0.99, goal_reward=10000):
        self.width = width
        self.height = height
        self.gamma = gamma
        self.actions = [0, 1, 2, 3, 4, 5, 6, 7] # Includes diagonal movements
        
        self.waypoints = waypoints
        self.num_phases = len(waypoints)
        
        # Logical states based on the number of phases
        self.states = [(x, y, q) for x in range(width) for y in range(height) for q in range(self.num_phases)]
        
        # The Final Goal is associated with the last phase index
        self.goal_state = (*self.waypoints[-1], self.num_phases - 1)
        self.goal_reward = goal_reward
        self.v_star = defaultdict(float)
        
    def get_transitions(self, state, action):
        x, y, q = state
        next_y = y
        reward = 0
        
        # Y-axis movement
        if action in [0, 4, 5]:    next_y = min(y + 1, self.height - 1)
        elif action in [1, 6, 7]:  next_y = max(y - 1, 0)
            
        # X-axis movement
        next_x = x
        if action in [2, 4, 6]:    next_x = max(x - 1, 0)
        elif action in [3, 5, 7]:  next_x = min(x + 1, self.width - 1)
        
        next_q = q
        
        # Dynamic phase transition logic: 
        # If the target of the current phase is reached, and we aren't at the end, advance the phase
        if q < self.num_phases - 1:
            target_x, target_y = self.waypoints[q]
            if x == target_x and y == target_y:
                next_q = q + 1

        next_state = (next_x, next_y, next_q)
        return next_state, reward

    def value_iteration(self, theta=0.001):
        print(f"Solving Sequential MDP ({self.num_phases} Phases) with Value Iteration...")
        self.v_star[self.goal_state] = self.goal_reward
        
        while True:
            delta = 0
            new_v = self.v_star.copy()
            for s in self.states:
                if s != self.goal_state: 
                    v_actions = [self.get_transitions(s, a)[1] + self.gamma * self.v_star[self.get_transitions(s, a)[0]] for a in self.actions]
                    best_v = max(v_actions)
                    delta = max(delta, abs(best_v - self.v_star[s]))
                    new_v[s] = best_v
            self.v_star = new_v
            if delta < theta: break