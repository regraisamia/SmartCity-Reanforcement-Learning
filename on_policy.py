import numpy as np
import matplotlib.pyplot as plt
from env_smartcity import SmartCityEnv
import time

def train_sarsa():
    # Initialisation de l'environnement Gymnasium [cite: 10]
    env = SmartCityEnv()
    
    # Initialisation du tableau Q (100 états x 4 actions) [cite: 10, 11]
    Q = np.zeros([100, 4])
    
    # Hyperparamètres
    alpha = 0.1    # Taux d'apprentissage
    gamma = 0.95   # Facteur d'actualisation
    epsilon = 0.2  # Taux d'exploration (Epsilon-greedy)
    episodes = 500 # Nombre d'itérations pour la convergence
    
    rewards_history = []
    start_time = time.time()

    print("Début de l'entraînement SARSA (On-Policy)...")

    for ep in range(episodes):
        state, _ = env.reset()
        
        # Choisir l'action initiale A selon la politique epsilon-greedy
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])
            
        total_reward = 0
        done = False
        
        # Boucle pour un épisode (limité à 100 étapes pour la stabilité)
        for _ in range(100):
            # Exécuter l'action A, observer R et l'état suivant S'
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # Choisir l'action suivante A' depuis S' (On-Policy car on suit la politique actuelle)
            if np.random.uniform(0, 1) < epsilon:
                next_action = env.action_space.sample()
            else:
                next_action = np.argmax(Q[next_state])
            
            # Mise à jour SARSA : Q(s,a) <- Q(s,a) + alpha * [R + gamma * Q(s',a') - Q(s,a)]
            target = reward + gamma * Q[next_state, next_action]
            Q[state, action] += alpha * (target - Q[state, action])
            
            # Transition vers l'état et l'action suivants
            state = next_state
            action = next_action
            total_reward += reward
            
            if terminated or truncated:
                break
            
        rewards_history.append(total_reward)

    # Calcul du temps d'exécution pour la comparaison 
    execution_time = time.time() - start_time
    
    print(f"--- Entraînement terminé ---")
    print(f"Temps d'exécution : {execution_time:.2f} secondes")
    print(f"Récompense moyenne finale : {np.mean(rewards_history[-50:]):.2f}")
    
    return rewards_history, Q

if __name__ == "__main__":
    history, q_table = train_sarsa()
    
    # Affichage de la courbe de convergence pour le rapport [cite: 23, 25]
    plt.figure(figsize=(10, 5))
    plt.plot(history, color='blue', label='SARSA (On-Policy)')
    plt.title("Évolution de la récompense cumulée (Convergence)")
    plt.xlabel("Épisodes")
    plt.ylabel("Récompense totale")
    plt.legend()
    plt.grid(True)
    plt.show()