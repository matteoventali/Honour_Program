import re
from collections import defaultdict
import numpy as np

# Importiamo il parser corretto dalla libreria ltlf2dfa
from ltlf2dfa.parser.ltlf import LTLfParser
from graphviz import Source

class LTLfAutomaton:
    """
    Wrapper per la libreria ltlf2dfa.
    Converte una formula LTLf in un DFA (formato DOT) e ne fa il parsing 
    in un grafo navigabile per l'MDP.
    """
    def __init__(self, formula_str):
        self.formula_str = formula_str
        
        # 1. Parsing della formula e generazione del DFA (in formato DOT)
        parser = LTLfParser()
        parsed_formula = parser(formula_str)
        dot_string = parsed_formula.to_dfa()
        self.dot_string = parsed_formula.to_dfa()
        
        # 2. Strutture dati dell'automa
        self.states = set()
        self.accepting_states = set()
        self.transitions = {}  # {stato_sorgente: [(condizione_booleana, stato_destinazione), ...]}
        self.initial_state = None
        
        # 3. Estrazione delle informazioni dalla stringa DOT
        self._parse_dot(dot_string)
        
        # Ordiniamo gli stati in una lista per l'MDP
        self.states = sorted(list(self.states))
        self.num_phases = len(self.states)

    def _parse_dot(self, dot_string):
        """
        Analizza la stringa DOT generata da ltlf2dfa ed estrae stati, 
        stati accettanti, stato iniziale e transizioni logiche.
        """
        # Estrazione degli stati accettanti (es. node [shape = doublecircle]; 2 3;)
        match_acc = re.search(r'node\s*\[shape\s*=\s*doublecircle\]\s*;\s*(.*?);', dot_string)
        if match_acc:
            acc_str = match_acc.group(1).replace(',', ' ')
            self.accepting_states = set(int(s) for s in acc_str.split() if s.strip().isdigit())
            
        # Estrazione delle transizioni (es. 1 -> 2 [label="wp1 & ~wp2"])
        trans_matches = re.findall(r'(\d+)\s*->\s*(\d+)\s*\[label\s*=\s*"(.*?)"\]', dot_string)
        for src_str, dst_str, guard in trans_matches:
            src = int(src_str)
            dst = int(dst_str)
            self.states.add(src)
            self.states.add(dst)
            
            if src not in self.transitions:
                self.transitions[src] = []
            self.transitions[src].append((guard, dst))
            
        # Estrazione dello stato iniziale (solitamente indicato da un arco senza etichetta da un nodo fantasma '0')
        # Es. 0 [style=invis]; 0 -> 1;
        init_match = re.search(r'(\d+)\s*->\s*(\d+)\s*;', dot_string)
        if init_match:
            self.initial_state = int(init_match.group(2))
        else:
            self.initial_state = min(self.states) if self.states else 0

    def get_initial_q(self):
        """Restituisce l'ID dello stato iniziale dell'automa."""
        return self.initial_state

    def is_goal_reached(self, current_q):
        """Verifica se lo stato attuale è uno stato accettante."""
        return current_q in self.accepting_states

    def get_next_q(self, current_q, truth_assignment):
        """
        Valuta le condizioni logiche (guardie) delle transizioni in uscita dallo 
        stato corrente e restituisce il prossimo stato dell'automa.
        """
        if current_q not in self.transitions:
            return current_q
            
        for guard, next_q in self.transitions[current_q]:
            if self._eval_guard(guard, truth_assignment):
                return next_q
                
        return current_q

    def _eval_guard(self, guard, truth_assignment):
        """
        Converte una guardia dal formato DOT (es. "wp1 & ~wp2") in Python 
        e la valuta rispetto al dizionario di verità attuale.
        """
        guard = guard.strip()
        
        # 1. Intercettiamo le costanti universali (sia formati numerici che testuali)
        if guard.lower() in ["1", "true"]: return True
        if guard.lower() in ["0", "false"]: return False
        
        # Mappiamo gli operatori logici standard in sintassi Python
        expr = guard.replace('&', ' and ').replace('|', ' or ').replace('~', ' not ').replace('!', ' not ')
        
        try:
            # 2. FIX: Usare un dizionario vuoto {} al posto di None per i builtins
            return eval(expr, {"__builtins__": {}}, truth_assignment)
        except Exception as e:
            print(f"[Errore LTLfAutomaton] Impossibile valutare la transizione '{guard}': {e}")
            return False

    def render_graph(self, filename="ltlf_automaton", directory="img"):
        """
        Renderizza e salva il DFA come immagine PNG.
        """
        try:
            src = Source(self.dot_string)
            src.render(filename=filename, directory=directory, format='png', cleanup=True)
            print(f"Grafo dell'automa salvato con successo in: {directory}/{filename}.png")
        except Exception as e:
            print(f"[Errore Graphviz] Impossibile renderizzare il grafo: {e}")


class LTLfWaypointMDP:
    """
    MDP per task sequenziali guidato da un automa LTLf.
    Lo stato astratto è 3D: (x, y, q) dove q è l'ID dello stato del DFA.
    """
    def __init__(self, waypoints_dict, ltlf_automaton, width=12, height=12, gamma=0.99, goal_reward=10000):
        self.width = width
        self.height = height
        self.gamma = gamma
        self.actions = [0, 1, 2, 3, 4, 5, 6, 7] # Include i movimenti diagonali
        
        self.waypoints_dict = waypoints_dict
        self.automaton = ltlf_automaton
        self.num_phases = self.automaton.num_phases
        
        # Generazione degli stati usando la griglia e tutti i possibili stati dell'automa
        self.states = [(x, y, q) for x in range(width) for y in range(height) for q in self.automaton.states]
        
        self.goal_reward = goal_reward
        self.v_star = defaultdict(float)
        
    def _get_truth_assignment(self, x, y):
        """
        Mappa le coordinate (x,y) attuali nelle proposizioni logiche.
        Restituisce un dizionario di verità per lo step dell'automa.
        """
        truth_assignment = {}
        for prop_name, (wp_x, wp_y) in self.waypoints_dict.items():
            truth_assignment[prop_name] = (x == wp_x and y == wp_y)
        return truth_assignment

    def get_transitions_old(self, state, action):
        x, y, q = state
        next_y = y
        reward = 0
        
        # Movimento asse Y
        if action in [0, 4, 5]:    next_y = min(y + 1, self.height - 1)
        elif action in [1, 6, 7]:  next_y = max(y - 1, 0)
            
        # Movimento asse X
        next_x = x
        if action in [2, 4, 6]:    next_x = max(x - 1, 0)
        elif action in [3, 5, 7]:  next_x = min(x + 1, self.width - 1)
        
        truth_assignment = self._get_truth_assignment(x, y)
        
        # Otteniamo il prossimo stato dell'automa dal modulo LTLf
        next_q = self.automaton.get_next_q(q, truth_assignment)
        next_state = (next_x, next_y, next_q)
        
        return next_state, reward

    def get_transitions(self, state, action):
        x, y, q = state
        reward = 0
        
        # 1. Calcola il movimento fisico NORMALE (come facevi prima)
        next_y = y
        if action in [0, 4, 5]:    next_y = min(y + 1, self.height - 1)
        elif action in [1, 6, 7]:  next_y = max(y - 1, 0)
            
        next_x = x
        if action in [2, 4, 6]:    next_x = max(x - 1, 0)
        elif action in [3, 5, 7]:  next_x = min(x + 1, self.width - 1)
        
        # 2. LA MODIFICA CHIAVE: Valuta le proposizioni sulle coordinate di ARRIVO
        # (Esempio generico, adatta alla tua funzione di valutazione)
        truth_assignment = self._get_truth_assignment(next_x, next_y)
        
        # 3. Aggiorna lo stato dell'automa basandoti su questa valutazione
        next_q = self.automaton.get_next_q(q, truth_assignment)

        next_state = (next_x, next_y, next_q)
        return next_state, reward
    
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