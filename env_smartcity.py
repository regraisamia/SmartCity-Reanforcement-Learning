#Ce code définit les règles du jeu, les 4 actions et la fonction de récompense.
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SmartCityEnv(gym.Env):
    def __init__(self):
        super(SmartCityEnv, self).__init__()
        
        # 1. ACTIONS : 4 choix possibles 
        # 0: NS Vert, 1: EO Vert, 2: NS Orange, 3: EO Orange
        self.action_space = spaces.Discrete(4)
        
        # 2. ÉTATS : 100 états distincts 
        # Représente (voitures_NS, voitures_EO) où chaque valeur est de 0 à 9
        self.observation_space = spaces.Discrete(100)
        
        self.state = 0 # État initial (0 voiture partout)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = 0 # On recommence à zéro
        return self.state, {}

    def step(self, action):
        cars_ns = self.state // 10
        cars_eo = self.state % 10

        # Logique de mouvement
        if action == 0: # NS Vert
            cars_ns = max(0, cars_ns - 2)
            cars_eo = min(9, cars_eo + 1)
        elif action == 1: # EO Vert
            cars_eo = max(0, cars_eo - 2)
            cars_ns = min(9, cars_ns + 1)

        self.state = cars_ns * 10 + cars_eo
        
        # --- JUSTIFICATION DE LA RÉCOMPENSE  ---
        # La récompense est R = -(cars_ns + cars_eo). 
        # Justification : Elle pénalise directement la congestion globale. 
        # L'agent doit maximiser R (viser 0) pour fluidifier le trafic.
        reward = -(cars_ns + cars_eo)
        
        return self.state, reward, False, False, {}
        cars_ns = self.state // 10
        cars_eo = self.state % 10

        if action == 0: # NS Vert
            cars_ns = max(0, cars_ns - 2)
            cars_eo = min(9, cars_eo + 1)
        elif action == 1: # EO Vert
            cars_eo = max(0, cars_eo - 2)
            cars_ns = min(9, cars_ns + 1)

        self.state = cars_ns * 10 + cars_eo
        
        # --- JUSTIFICATION DE LA RÉCOMPENSE (Critère PDF) ---
        # La récompense est négative pour pénaliser la congestion (somme des voitures).
        # L'agent cherche à maximiser cette valeur (la rapprocher de 0),
        # ce qui force mathématiquement la réduction des files d'attente.
        reward = -(cars_ns + cars_eo) 
        
        return self.state, reward, False, False, {}
        # On décompose l'état (ex: état 45 -> 4 voitures NS, 5 voitures EO)
        cars_ns = self.state // 10
        cars_eo = self.state % 10

        # Logique simplifiée : si le feu est vert, les voitures diminuent
        if action == 0: # NS Vert
            cars_ns = max(0, cars_ns - 2)
            cars_eo = min(9, cars_eo + 1) # De nouvelles voitures arrivent
        elif action == 1: # EO Vert
            cars_eo = max(0, cars_eo - 2)
            cars_ns = min(9, cars_ns + 1)
        # (Les actions Orange pourraient simplement ajouter des voitures)

        # Calcul du nouvel état
        self.state = cars_ns * 10 + cars_eo
        
        # 3. RÉCOMPENSE : On pénalise la congestion 
        # Moins il y a de voitures en attente, plus la récompense est haute.
        reward = -(cars_ns + cars_eo)
        
        terminated = False # Le trafic ne s'arrête jamais vraiment
        truncated = False
        
        return self.state, reward, terminated, truncated, {}