from env_smartcity import SmartCityEnv

# 1. Création de l'environnement
env = SmartCityEnv()

# 2. Réinitialisation
state, info = env.reset()
print(f"État initial : {state} (NS: {state//10}, EO: {state%10})")

# 3. Simulation de 10 étapes avec des actions aléatoires
print("\nDébut de la simulation de test :")
for i in range(1, 11):
    # Choisir une action au hasard parmi les 4 possibles
    action = env.action_space.sample() 
    
    # Appliquer l'action
    next_state, reward, terminated, truncated, info = env.step(action)
    
    # Affichage des résultats
    print(f"Étape {i} | Action choisie: {action} | Nouvel État: {next_state} | Récompense: {reward}")

print("\nTest terminé avec succès !")
