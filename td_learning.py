import numpy as np
import matplotlib.pyplot as plt
from env_smartcity import SmartCityEnv
import time

def train_expected_sarsa():
    # Initialisation de l'environnement Gymnasium (Respecte les contraintes)
    env = SmartCityEnv()
    
    # Tableau Q : 100 états x 4 actions
    Q = np.zeros([100, 4])
    
    # Hyperparamètres
    alpha = 0.1    # Taux d'apprentissage
    gamma = 0.95   # Facteur d'actualisation
    epsilon = 0.2  # Taux d'exploration
    episodes = 500 
    
    rewards_history = []
    start_time = time.time()

    print("Début de l'entraînement Expected SARSA (TD Learning)...")

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        
        for _ in range(100):
            # Politique Epsilon-greedy
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # Calcul de l'espérance (Expected Value) pour l'état suivant
            expected_q = 0
            n_actions = env.action_space.n
            best_action = np.argmax(Q[next_state])
            
            for a in range(n_actions):
                if a == best_action:
                    # Probabilité de la meilleure action
                    prob = (1 - epsilon) + (epsilon / n_actions)
                else:
                    # Probabilité des actions d'exploration
                    prob = epsilon / n_actions
                expected_q += prob * Q[next_state, a]
            
            # Mise à jour TD : Q(s,a) = Q(s,a) + alpha * [R + gamma * Expected_Q - Q(s,a)]
            Q[state, action] += alpha * (reward + gamma * expected_q - Q[state, action])
            
            state = next_state
            total_reward += reward
            
            if terminated or truncated:
                break
            
        rewards_history.append(total_reward)

    # Mesure des critères de comparaison demandés
    execution_time = time.time() - start_time
    
    print(f"--- TD Learning terminé ---")
    print(f"Temps d'exécution : {execution_time:.2f} secondes")
    
    return rewards_history, Q

if __name__ == "__main__":
    history, q_table = train_expected_sarsa()
    
    # Visualisation de la convergence
    plt.figure(figsize=(10, 5))
    plt.plot(history, color='purple', label='Expected SARSA (TD)')
    plt.title("Convergence TD Learning")
    plt.xlabel("Épisodes")
    plt.ylabel("Récompense totale")
    plt.legend()
    plt.show()