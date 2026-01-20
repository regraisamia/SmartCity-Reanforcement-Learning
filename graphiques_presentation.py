import matplotlib.pyplot as plt
import numpy as np
from on_policy import train_sarsa
from off_policy import train_q_learning
from td_learning import train_expected_sarsa
from approximation_fonction import train_dqn


plt.rcParams['font.size'] = 16
plt.rcParams['figure.figsize'] = (12, 8)

print("Génération des graphiques ...")

# Exécution rapide des algorithmes
hist_sarsa, _ = train_sarsa()
hist_q, _ = train_q_learning()
hist_td, _ = train_expected_sarsa()
hist_dqn = train_dqn()

# Données pour graphiques
resultats = {
    'SARSA': {'hist': hist_sarsa, 'temps': 1.47, 'perf': -105.38, 'stab': 29.83},
    'Q-Learning': {'hist': hist_q, 'temps': 1.44, 'perf': -105.70, 'stab': 37.06},
    'Expected SARSA': {'hist': hist_td, 'temps': 2.88, 'perf': -104.58, 'stab': 24.10},
    'DQN': {'hist': hist_dqn, 'temps': 113.22, 'perf': -97.20, 'stab': 11.62}
}

couleurs = ['#3498db', '#e74c3c', '#9b59b6', '#f39c12']

# GRAPHIQUE 1: Convergence 
plt.figure(figsize=(14, 8))
for i, (algo, data) in enumerate(resultats.items()):
    plt.plot(data['hist'], label=algo, linewidth=3, color=couleurs[i])

plt.title('Convergence des Algorithmes', fontsize=20, fontweight='bold')
plt.xlabel('Épisodes', fontsize=16)
plt.ylabel('Récompense Cumulée', fontsize=16)
plt.legend(fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('slide_convergence.png', dpi=300, bbox_inches='tight')
plt.show()

# GRAPHIQUE 2: Temps d'exécution 
plt.figure(figsize=(12, 8))
algos = list(resultats.keys())
temps = [resultats[algo]['temps'] for algo in algos]

bars = plt.bar(algos, temps, color=couleurs, alpha=0.8, edgecolor='black', linewidth=2)
plt.title('Temps d\'Exécution', fontsize=20, fontweight='bold')
plt.ylabel('Secondes', fontsize=16)
plt.xticks(rotation=45, fontsize=14)

# Ajout des valeurs sur les barres
for bar, val in zip(bars, temps):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(temps)*0.02, 
             f'{val:.1f}s', ha='center', va='bottom', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('slide_temps.png', dpi=300, bbox_inches='tight')
plt.show()

# GRAPHIQUE 3: Performance finale
plt.figure(figsize=(12, 8))
performances = [resultats[algo]['perf'] for algo in algos]

bars = plt.bar(algos, performances, color=couleurs, alpha=0.8, edgecolor='black', linewidth=2)
plt.title('Performance Finale', fontsize=20, fontweight='bold')
plt.ylabel('Récompense Moyenne', fontsize=16)
plt.xticks(rotation=45, fontsize=14)

# Ajout des valeurs
for bar, val in zip(bars, performances):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + abs(min(performances))*0.01, 
             f'{val:.1f}', ha='center', va='bottom', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('slide_performance.png', dpi=300, bbox_inches='tight')
plt.show()

# GRAPHIQUE 4: Stabilité 
plt.figure(figsize=(12, 8))
stabilites = [resultats[algo]['stab'] for algo in algos]

bars = plt.bar(algos, stabilites, color=couleurs, alpha=0.8, edgecolor='black', linewidth=2)
plt.title('Stabilité (Écart-type)', fontsize=20, fontweight='bold')
plt.ylabel('Écart-type', fontsize=16)
plt.xticks(rotation=45, fontsize=14)

# Ajout des valeurs
for bar, val in zip(bars, stabilites):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(stabilites)*0.02, 
             f'{val:.1f}', ha='center', va='bottom', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('slide_stabilite.png', dpi=300, bbox_inches='tight')
plt.show()

print(" Graphiques  générés !")
print(" Fichiers créés:")
print("   - slide_convergence.png")
print("   - slide_temps.png") 
print("   - slide_performance.png")
print("   - slide_stabilite.png")