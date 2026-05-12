from collections import defaultdict

class AbstractGridMDP:
    def __init__(self, width=12, height=12, gamma=0.99):
        self.width = width
        self.height = height
        self.gamma = gamma
        self.states = [(x, y) for x in range(width) for y in range(height)]
        self.actions = [0, 1, 2, 3]  
        self.goal_state = (width // 2, 0)
        self.v_star = defaultdict(float)

    def get_transitions(self, state, action):
        x, y = state
        if action == 0: y = min(y + 1, self.height - 1) 
        elif action == 1: y = max(y - 1, 0)             
        elif action == 2: x = max(x - 1, 0)             
        elif action == 3: x = min(x + 1, self.width - 1) 
        
        next_state = (x, y)
        reward = 1.0 if next_state == self.goal_state else 0.0
        return next_state, reward

    def value_iteration(self, theta=0.001):
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

class DiagonalAbstractGridMDP(AbstractGridMDP):
    def __init__(self, width=12, height=12, gamma=0.99):
        super().__init__(width, height, gamma)
        self.actions = [0, 1, 2, 3, 4, 5, 6, 7]

    def get_transitions(self, state, action):
        x, y = state
        if action in [0, 4, 5]:    y = min(y + 1, self.height - 1)
        elif action in [1, 6, 7]:  y = max(y - 1, 0)
        if action in [2, 4, 6]:    x = max(x - 1, 0)
        elif action in [3, 5, 7]:  x = min(x + 1, self.width - 1)
            
        next_state = (x, y)
        reward = 1.0 if next_state == self.goal_state else 0.0
        return next_state, reward

class KinematicAbstractMDP:
    def __init__(self, width=12, height=12, gamma=0.99):
        self.width = width
        self.height = height
        self.gamma = gamma
        self.states = [(x, y, vy, a) for x in range(width) for y in range(height) for vy in range(3) for a in range(3)]
        self.actions = [0, 1, 2, 3]  
        self.goal_state = (width // 2, 0, 1, 1) # Center, safe speed, straight angle
        self.v_star = defaultdict(float)

    def get_transitions(self, state, action):
        x, y, vy, a = state
        next_vy, next_a, next_x, next_y = vy, a, x, y
        
        if action == 0: 
            next_y = max(y - 1, 0)
            next_vy = 0 
        elif action == 2: 
            next_y = min(y + 1, self.height - 1)
            next_vy = 2 
        elif action == 1: 
            next_x = min(x + 1, self.width - 1)
            next_a = 0
            next_y = max(y - 1, 0) 
        elif action == 3: 
            next_x = max(x - 1, 0)
            next_a = 2
            next_y = max(y - 1, 0) 
            
        next_state = (next_x, next_y, next_vy, next_a)
        reward = 1.0 if next_state == self.goal_state else 0.0
        return next_state, reward

    def value_iteration(self, theta=0.001):
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