import numpy as np
import matplotlib.pyplot as plt
from env_smartcity import SmartCityEnv
import time

def train_q_learning():
    # Initialisation de l'environnement Gymnasium (Min 100 états) [cite: 10]
    env = SmartCityEnv()
    
    # Initialisation du tableau Q (100 états x 4 actions) [cite: 10, 11]
    Q = np.zeros([100, 4])
    
    # Hyperparamètres
    alpha = 0.1    # Taux d'apprentissage
    gamma = 0.95   # Facteur d'actualisation
    epsilon = 0.2  # Taux d'exploration
    episodes = 500 # Itérations pour la convergence 
    
    rewards_history = []
    start_time = time.time()

    print("Début de l'entraînement Q-Learning (Off-Policy)...")

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        
        for _ in range(100):
            # Choisir l'action A selon la politique epsilon-greedy
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])
            
            # Exécuter l'action A, observer R et l'état suivant S'
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # MISE À JOUR Q-LEARNING (Off-Policy) 
            # On utilise le MAX des actions possibles pour l'état suivant
            best_next_action = np.argmax(Q[next_state])
            target = reward + gamma * Q[next_state, best_next_action]
            Q[state, action] += alpha * (target - Q[state, action])
            
            state = next_state
            total_reward += reward
            
            if terminated or truncated:
                break
            
        rewards_history.append(total_reward)

    # Mesure du temps d'exécution pour la comparaison 
    execution_time = time.time() - start_time
    
    print(f"--- Entraînement terminé ---")
    print(f"Temps d'exécution : {execution_time:.2f} secondes")
    print(f"Récompense moyenne finale : {np.mean(rewards_history[-50:]):.2f}")
    
    return rewards_history, Q

if __name__ == "__main__":
    # Appel de la fonction correctement nommée
    history, q_table = train_q_learning()
    
    # Graphique pour l'analyse de la convergence [cite: 23, 25]
    plt.figure(figsize=(10, 5))
    plt.plot(history, color='orange', label='Q-Learning (Off-Policy)')
    plt.title("Convergence Q-Learning (Off-Policy)")
    plt.xlabel("Épisodes")
    plt.ylabel("Récompense totale")
    plt.legend()
    plt.grid(True)
    plt.show()