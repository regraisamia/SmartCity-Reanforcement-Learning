import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from env_smartcity import SmartCityEnv
import time
import matplotlib.pyplot as plt

# 1. Définition du Réseau de Neurones (Approximation de fonction)
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.fc(x)

def train_dqn():
    env = SmartCityEnv()
    # On transforme l'état discret (0-99) en un vecteur pour le réseau
    state_dim = 1 
    action_dim = env.action_space.n
    
    model = DQN(state_dim, action_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    gamma = 0.95
    epsilon = 0.2
    episodes = 200 # Moins d'épisodes car c'est plus lent
    rewards_history = []
    start_time = time.time()

    print("Début de l'entraînement DQN (Approximation de fonction)...")

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        
        for _ in range(100):
            # Conversion de l'état en tenseur
            state_t = torch.FloatTensor([state])
            
            # Epsilon-greedy
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = model(state_t).argmax().item()
            
            next_state, reward, _, _, _ = env.step(action)
            
            # Préparation de la cible (Target)
            state_t = torch.FloatTensor([state])
            next_state_t = torch.FloatTensor([next_state])
            
            current_q = model(state_t)[action]
            with torch.no_grad():
                max_next_q = model(next_state_t).max()
                target_q = reward + gamma * max_next_q
            
            # Mise à jour des poids du réseau
            loss = criterion(current_q, target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            state = next_state
            total_reward += reward
            
        rewards_history.append(total_reward)

    execution_time = time.time() - start_time
    print(f"--- DQN terminé ---")
    print(f"Temps d'exécution : {execution_time:.2f} secondes")
    return rewards_history

if __name__ == "__main__":
    history = train_dqn()
    plt.plot(history, color='red', label='DQN (Approximation)')
    plt.title("Convergence DQN")
    plt.legend()
    plt.show()