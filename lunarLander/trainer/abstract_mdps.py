from collections import defaultdict
import numpy as np

class AbstractGridMDP:
    def __init__(self, width=12, height=12, gamma=0.99):
        self.width = width
        self.height = height
        self.gamma = gamma
        self.states = [(x, y) for x in range(width) for y in range(height)]
        self.actions = [0, 1, 2, 3]  
        
        # Goal 2D spostato a sinistra per l'esperimento di debugging
        #self.goal_state = (0, height // 2)

        center_continuous_x = 0.0
        mapped_center_x = int(np.clip((center_continuous_x + 1) / 2 * (width - 1), 0, width - 1))
        self.goal_state = (mapped_center_x, 0)

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
        
        # Goal cinematico a sinistra per esperimento di shaping
        self.goal_state = (0, 0, 1, 1)

        #center_continuous_x = 0.0
        #mapped_center_x = int(np.clip((center_continuous_x + 1) / 2 * (width - 1), 0, width - 1))
        #
        ## Impostiamo il goal_state dinamicamente
        #self.goal_state = (mapped_center_x, 0, 1, 1)

        self.v_star = defaultdict(float)

    def get_transitions(self, state, action):
        x, y, vy, a = state

        # ---- NUOVA MODIFICA: LA TRAPPOLA PER LA SALITA ----
        # Se l'agente sta salendo (vy == 2), lo consideriamo un "buco nero" 
        # (sink state). L'agente astratto rimane bloccato qui con reward 0.0.
        # Questo costringerà il Value Iteration ad assegnare un potenziale 
        # ESATTAMENTE PARI A 0.0 a tutti gli stati in cui il lander sale.
        if vy == 2:
            return state, 0.0

        next_x = x
        next_y = y
        next_vy = vy
        next_a = a

        # Azioni cinematiche astratte
        if action == 0: # Nessuna spinta
            if y > 0: next_y -= 1
            next_vy = 0
            
        elif action == 1: # Spinta motore sinistro (sposta a destra)
            if x < self.width - 1: next_x += 1
            
            # MODIFICA: Variazione incrementale dell'angolo
            if a == 2: next_a = 1   # Se era inclinato a dx, si raddrizza
            else: next_a = 0        # Altrimenti si inclina a sx
            
            # (Mantieni la logica del veleggiamento se l'avevi inserita)
            if y > 0 and vy == 0: next_y -= 1 
            
        elif action == 2: # Spinta motore principale
            if y > 0: next_y -= 1
            next_vy = 1
            # MODIFICA: Il motore principale NON raddrizza più l'angolo.
            # Rimuovi "next_a = 1", l'angolo rimane quello che era (next_a = a)
            
        elif action == 3: # Spinta motore destro (sposta a sinistra)
            if x > 0: next_x -= 1
            
            # MODIFICA: Variazione incrementale dell'angolo
            if a == 0: next_a = 1   # Se era inclinato a sx, si raddrizza
            else: next_a = 2        # Altrimenti si inclina a dx
            
            if y > 0 and vy == 0: next_y -= 1

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