from collections import defaultdict
import numpy as np

class AbstractGridMDP:
    """
    Standard 2D Grid Abstraction for the LunarLander.
    The agent can only move Up, Down, Left, or Right (no diagonals).
    State representation: (X, Y)
    """
    def __init__(self, width=12, height=12, gamma=0.99):
        self.width = width
        self.height = height
        self.gamma = gamma
        self.states = [(x, y) for x in range(width) for y in range(height)]
        self.actions = [0, 1, 2, 3]  
        
        # 2D Goal moved to the bottom-left for the debugging experiment
        self.goal_state = (0, 0)

        # Optional: Dynamic center mapping (commented out for current experiments)
        # center_continuous_x = 0.0
        # mapped_center_x = int(np.clip((center_continuous_x + 1) / 2 * (width - 1), 0, width - 1))
        # self.goal_state = (mapped_center_x, 0)

        self.v_star = defaultdict(float)

    def get_transitions(self, state, action):
        x, y = state
        
        # Standard Gridworld Movement
        if action == 0: y = min(y + 1, self.height - 1)  # Move Up
        elif action == 1: y = max(y - 1, 0)              # Move Down
        elif action == 2: x = max(x - 1, 0)              # Move Left
        elif action == 3: x = min(x + 1, self.width - 1) # Move Right
        
        next_state = (x, y)
        reward = 1.0 if next_state == self.goal_state else 0.0
        return next_state, reward

    def value_iteration(self, theta=0.001):
        """Standard Value Iteration algorithm to compute V*."""
        print("Solving Abstract MDP with Value Iteration...")
        while True:
            delta = 0
            new_v = self.v_star.copy()
            for s in self.states:
                if s == self.goal_state: continue
                v_actions = [self.get_transitions(s, a)[1] + self.gamma * self.v_star[self.get_transitions(s, a)[0]] for a in self.actions]
                best_v = max(v_actions)
                delta = max(delta, abs(best_v - self.v_star[s]))
                new_v[s] = best_v
            self.v_star = new_v
            if delta < theta: break
        self.v_star[self.goal_state] = 1.0


class DiagonalAbstractGridMDP(AbstractGridMDP):
    """
    Enhanced 2D Grid Abstraction.
    Allows the agent to move in 8 directions (including diagonals),
    enabling smoother, continuous trajectories for shaping.
    """
    def __init__(self, width=12, height=12, gamma=0.99):
        super().__init__(width, height, gamma)
        # Actions 0-3: Standard (Up, Down, Left, Right)
        # Actions 4-7: Diagonals (Up-Left, Up-Right, Down-Left, Down-Right)
        self.actions = [0, 1, 2, 3, 4, 5, 6, 7]

    def get_transitions(self, state, action):
        x, y = state
        
        # Calculate Y-axis changes
        if action in [0, 4, 5]:    y = min(y + 1, self.height - 1)
        elif action in [1, 6, 7]:  y = max(y - 1, 0)
            
        # Calculate X-axis changes
        if action in [2, 4, 6]:    x = max(x - 1, 0)
        elif action in [3, 5, 7]:  x = min(x + 1, self.width - 1)
            
        next_state = (x, y)
        reward = 1.0 if next_state == self.goal_state else 0.0
        return next_state, reward


class ConfigurableDiagonalMDP(DiagonalAbstractGridMDP):
    """
    A configurable version of the DiagonalAbstractGridMDP designed specifically 
    for Grid Search and Hyperparameter tuning. It supports dynamic goal sets, 
    variable gamma values, and adjustable reward magnitudes.
    """
    def __init__(self, width=12, height=12, gamma=0.99, goal_states=[(0,0)], goal_reward=1.0):
        super().__init__(width, height, gamma)
        # Replaces the single goal_state with a Set of multiple valid goal_states
        self.goal_states = set(goal_states)
        self.goal_reward = goal_reward

    def get_transitions(self, state, action):
        # Inherit diagonal movement logic from parent
        next_state, _ = super().get_transitions(state, action)
        
        # Reward is granted if the agent enters ANY of the designated goal cells
        reward = self.goal_reward if next_state in self.goal_states else 0.0
        return next_state, reward

    def value_iteration(self, theta=0.001):
        """Updated Value Iteration to support multiple goal states and variable rewards."""
        while True:
            delta = 0
            new_v = self.v_star.copy()
            for s in self.states:
                # Skip computation if the state is one of our goals
                if s in self.goal_states: continue
                
                v_actions = [self.get_transitions(s, a)[1] + self.gamma * self.v_star[self.get_transitions(s, a)[0]] for a in self.actions]
                best_v = max(v_actions)
                delta = max(delta, abs(best_v - self.v_star[s]))
                new_v[s] = best_v
                
            self.v_star = new_v
            if delta < theta: break
            
        # Ensure all goal states strictly hold their maximum reward value
        for g in self.goal_states:
            self.v_star[g] = self.goal_reward


class ValleyDiagonalAbstractMDP(DiagonalAbstractGridMDP):
    """
    Experiment 3 Variant: The Potential Valley.
    Instead of manually zeroing out states after computation, this MDP 
    injects a transition cost (-1.0) for leaving the desired trajectory.
    This mathematically carves a V* gradient (a funnel) pointing towards the path.
    """
    def __init__(self, trajectory_path, width=12, height=12, gamma=0.99):
        super().__init__(width, height, gamma)
        # Convert the trajectory list to a Set for O(1) instant lookups
        self.trajectory_path = set(trajectory_path)

    def get_transitions(self, state, action):
        # Inherit standard movement calculation
        next_state, base_reward = super().get_transitions(state, action)
        
        # If it hits the final goal, return the clean base reward
        if next_state == self.goal_state:
            return next_state, base_reward
            
        # THE MAGIC OF THE VALLEY: We inject a negative transition cost 
        # if the destination state IS NOT part of our desired trajectory.
        if next_state not in self.trajectory_path:
            step_penalty = -1.0  # Digs the potential downward (creates the valley)
        else:
            step_penalty = 0.0   # The golden path costs nothing
            
        total_reward = base_reward + step_penalty
        return next_state, total_reward


class KinematicAbstractMDP:
    """
    4D Kinematic Abstraction.
    Introduces physics (velocity and angle) directly into the abstract model.
    State representation: (X, Y, Velocity_Y, Angle)
    """
    def __init__(self, width=12, height=12, gamma=0.99):
        self.width = width
        self.height = height
        self.gamma = gamma
        
        # State space expansion: x, y, vertical_velocity (3 states), angle (3 states)
        self.states = [(x, y, vy, a) for x in range(width) for y in range(height) for vy in range(3) for a in range(3)]
        self.actions = [0, 1, 2, 3]  
        
        # Kinematic goal on the bottom-left for the shaping experiment
        # Goal state format: (X=0, Y=0, Velocity_Y=1 (slow/safe fall), Angle=1 (upright))
        self.goal_state = (0, 0, 1, 1)

        self.v_star = defaultdict(float)

    def get_transitions(self, state, action):
        x, y, vy, a = state

        # ---- THE ASCENT TRAP (SINK STATE) ----
        # If the agent is moving upward (vy == 2), we treat it as a "black hole".
        # The abstract agent gets stuck here with a 0.0 reward.
        # This elegantly forces the Value Iteration to assign a potential of 
        # EXACTLY 0.0 to all states where the lander goes up, discouraging climbing.
        if vy == 2:
            return state, 0.0

        next_x = x
        next_y = y
        next_vy = vy
        next_a = a

        # Abstract kinematic actions mapping to LunarLander controls
        if action == 0: # Do nothing (No engine)
            if y > 0: next_y -= 1
            next_vy = 0
            
        elif action == 1: # Fire Left Engine (Pushes lander to the Right)
            if x < self.width - 1: next_x += 1
            
            # Incremental angle variation
            if a == 2: next_a = 1   # If tilted right, firing left engine straightens it
            else: next_a = 0        # Otherwise, it tilts further to the left
            
            # Gravity still acts if not thrusting main engine
            if y > 0 and vy == 0: next_y -= 1 
            
        elif action == 2: # Fire Main Engine (Pushes Upward / Slows Fall)
            if y > 0: next_y -= 1
            next_vy = 1
            # The main engine does not straighten the angle.
            # The angle remains whatever it currently is (next_a = a)
            
        elif action == 3: # Fire Right Engine (Pushes lander to the Left)
            if x > 0: next_x -= 1
            
            # Incremental angle variation
            if a == 0: next_a = 1   # If tilted left, firing right engine straightens it
            else: next_a = 2        # Otherwise, it tilts further to the right
            
            if y > 0 and vy == 0: next_y -= 1

        next_state = (next_x, next_y, next_vy, next_a)
        reward = 1.0 if next_state == self.goal_state else 0.0
        return next_state, reward

    def value_iteration(self, theta=0.001):
        """Value iteration for the larger 4D state space."""
        print(f"Solving 4D Kinematic Abstract MDP ({len(self.states)} states)...")
        while True:
            delta = 0
            new_v = self.v_star.copy()
            
            for s in self.states:
                if s == self.goal_state: continue
                v_actions = [self.get_transitions(s, act)[1] + self.gamma * self.v_star[self.get_transitions(s, act)[0]] for act in self.actions]
                best_v = max(v_actions)
                delta = max(delta, abs(best_v - self.v_star[s]))
                new_v[s] = best_v

            self.v_star = new_v
            if delta < theta: break
        self.v_star[self.goal_state] = 1.0


class SequentialWaypointMDP:
    """
    MDP for sequential tasks.
    The abstract state is 3D: (x, y, q)
    q = 0: searching for the waypoint (1, 8)
    q = 1: searching for the final goal (8, 8)
    """
    def __init__(self, width=12, height=12, gamma=0.99):
        self.width = width
        self.height = height
        self.gamma = gamma
        self.actions = [0, 1, 2, 3, 4, 5, 6, 7]
        self.states = [(x, y, q) for x in range(width) for y in range(height) for q in (0, 1)]
        self.waypoint = (1, 8)
        self.goal_state = (8, 8, 1) 
        self.v_star = defaultdict(float)
        self.waypoint_reached = False

    def get_transitions(self, state, action):
        x, y, q = state
        next_y = y
        if action in [0, 4, 5]:    next_y = min(y + 1, self.height - 1)
        elif action in [1, 6, 7]:  next_y = max(y - 1, 0)
            
        next_x = x
        if action in [2, 4, 6]:    next_x = max(x - 1, 0)
        elif action in [3, 5, 7]:  next_x = min(x + 1, self.width - 1)
        
        next_q = q
        #if next_x == self.waypoint[0] and next_y == self.waypoint[1] and q == 0:
        #    self.waypoint_reached = True
        #elif self.waypoint_reached and q == 0:
        #    self.waypoint_reached = False
        #    next_q = 1

        if x == self.waypoint[0] and y == self.waypoint[1] and next_q == 0:
            next_q = 1

        next_state = (next_x, next_y, next_q)
        reward = 100.0 if next_state == self.goal_state else 0.0
        return next_state, reward

    def value_iteration(self, theta=0.001):
        print("Solving Sequential MDP with Value Iteration...")
        while True:
            delta = 0
            new_v = self.v_star.copy()
            for s in self.states:
                if s == self.goal_state: 
                    continue
                v_actions = [self.get_transitions(s, a)[1] + self.gamma * self.v_star[self.get_transitions(s, a)[0]] for a in self.actions]
                best_v = max(v_actions)
                delta = max(delta, abs(best_v - self.v_star[s]))
                new_v[s] = best_v
            self.v_star = new_v
            if delta < theta: break
        
        self.v_star[self.goal_state] = 100.0
        