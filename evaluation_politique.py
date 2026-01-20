import numpy as np
from env_smartcity import SmartCityEnv
from off_policy import train_q_learning

def evaluate_agent(env, q_table, episodes=100):
    total_rewards = []
    for _ in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        for _ in range(100):
            action = np.argmax(q_table[state]) # Politique optimale
            state, reward, _, _, _ = env.step(action)
            episode_reward += reward
        total_rewards.append(episode_reward)
    return np.mean(total_rewards)

def evaluate_random(env, episodes=100):
    total_rewards = []
    for _ in range(episodes):
        env.reset()
        episode_reward = 0
        for _ in range(100):
            action = env.action_space.sample() # Action au hasard
            _, reward, _, _, _ = env.step(action)
            episode_reward += reward
        total_rewards.append(episode_reward)
    return np.mean(total_rewards)

if __name__ == "__main__":
    env = SmartCityEnv()
    print("Évaluation de la qualité de la politique...")
    _, q_table = train_q_learning()
    
    score_optimal = evaluate_agent(env, q_table)
    score_random = evaluate_random(env)
    
    print(f"\n--- RÉSULTATS D'ÉVALUATION ---")
    print(f"Récompense moyenne (Agent IA) : {score_optimal:.2f}")
    print(f"Récompense moyenne (Agent Aléatoire) : {score_random:.2f}")
    print(f"Amélioration : {abs(((score_optimal - score_random) / score_random) * 100):.2f}%")