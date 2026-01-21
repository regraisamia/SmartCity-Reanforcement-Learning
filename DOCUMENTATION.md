# Documentation du Projet - Smart City Traffic Control

##  Architecture du Projet

### Fichiers Principaux

| Fichier | Type | Description |
|---------|------|-------------|
| `env_smartcity.py` | Environnement | Intersection urbaine avec Gymnasium |
| `on_policy.py` | Algorithme | SARSA (On-Policy) |
| `off_policy.py` | Algorithme | Q-Learning (Off-Policy) |
| `td_learning.py` | Algorithme | Expected SARSA (TD Learning) |
| `approximation_fonction.py` | Algorithme | DQN (Approximation de fonction) |
| `comparaison_finale.py` | Analyse | Comparaison des 4 algorithmes |
| `evaluation_politique.py` | Évaluation | Test de qualité des politiques |
| `analyse_comparative.ipynb` | Notebook | Résultats détaillés avec graphiques |

---

##  Algorithmes Implémentés

### 1. SARSA - On-Policy (`on_policy.py`)

**Principe :** Suit la politique actuelle pendant l'apprentissage

**Mise à jour :**
```
Q(s,a) ← Q(s,a) + α[R + γQ(s',a') - Q(s,a)]
```

**Paramètres :**
- `alpha = 0.1` - Taux d'apprentissage
- `gamma = 0.95` - Facteur d'actualisation
- `epsilon = 0.2` - Taux d'exploration
- `episodes = 500` - Nombre d'épisodes

**Caractéristiques :**
- Conservateur (suit sa propre politique)
- Stable mais peut être sous-optimal

---

### 2. Q-Learning - Off-Policy (`off_policy.py`)

**Principe :** Apprend la politique optimale indépendamment de la politique suivie

**Mise à jour :**
```
Q(s,a) ← Q(s,a) + α[R + γ max Q(s',a) - Q(s,a)]
```

**Paramètres :**
- `alpha = 0.1` - Taux d'apprentissage
- `gamma = 0.95` - Facteur d'actualisation
- `epsilon = 0.2` - Taux d'exploration
- `episodes = 500` - Nombre d'épisodes

**Caractéristiques :**
- Optimiste (vise la meilleure action)
- Convergence vers politique optimale

---

### 3. Expected SARSA - TD Learning (`td_learning.py`)

**Principe :** Calcule l'espérance des Q-values au lieu du maximum

**Mise à jour :**
```
Q(s,a) ← Q(s,a) + α[R + γ E[Q(s',a')] - Q(s,a)]
```

**Calcul de l'espérance :**
```python
expected_q = Σ π(a'|s') × Q(s',a')
```

**Paramètres :**
- `alpha = 0.1` - Taux d'apprentissage
- `gamma = 0.95` - Facteur d'actualisation
- `epsilon = 0.2` - Taux d'exploration
- `episodes = 500` - Nombre d'épisodes

**Caractéristiques :**
- Combine avantages on-policy et off-policy
- Plus stable que Q-Learning

---

### 4. DQN - Approximation de Fonction (`approximation_fonction.py`)

**Principe :** Utilise un réseau de neurones pour approximer Q(s,a)

**Architecture du réseau :**
```
Input(1) → Dense(64) → ReLU → Dense(64) → ReLU → Output(4)
```

**Paramètres :**
- `lr = 0.001` - Learning rate (Adam)
- `gamma = 0.95` - Facteur d'actualisation
- `epsilon = 0.2` - Taux d'exploration
- `episodes = 200` - Moins d'épisodes (plus lent)

**Caractéristiques :**
- Gère les grands espaces d'états
- Plus lent mais plus expressif

---

## Environnement SmartCity (`env_smartcity.py`)

### Spécifications Gymnasium

```python
observation_space = spaces.Discrete(100)  # 100 états distincts
action_space = spaces.Discrete(4)         # 4 actions possibles
```

### États (100 possibles)
- **Encodage :** `état = (voitures_NS × 10) + voitures_EO`
- **Exemple :** État 45 = 4 voitures Nord-Sud + 5 voitures Est-Ouest
- **Plage :** 0-99 (0-9 voitures par axe)

### Actions (4 possibles)
- **Action 0 :** Feu VERT Nord-Sud
- **Action 1 :** Feu VERT Est-Ouest
- **Action 2 :** Feu ORANGE Nord-Sud
- **Action 3 :** Feu ORANGE Est-Ouest

### Dynamique du Système
```python
# Feu VERT → Évacue 2 voitures, 1 nouvelle arrive sur l'autre axe
if action == 0:  # NS Vert
    cars_ns = max(0, cars_ns - 2)
    cars_eo = min(9, cars_eo + 1)
```

### Fonction de Récompense
```python
reward = -(cars_ns + cars_eo)
```
**Justification :** Pénalise la congestion totale. Plus proche de 0 = meilleur.

---

##  Critères de Comparaison

### Métriques Évaluées
1. **Temps d'exécution** - Rapidité d'entraînement
2. **Performance finale** - Récompense moyenne (50 derniers épisodes)
3. **Stabilité** - Écart-type des performances
4. **Convergence** - Vitesse d'amélioration
5. **Politique optimale** - Qualité de la stratégie apprise

### Résultats Typiques
| Algorithme | Performance | Temps | Stabilité |
|------------|-------------|-------|-----------|
| SARSA | -103.98 | 0.69s | Moyenne |
| Q-Learning | -105.44 | 0.97s | Faible |
| Expected SARSA | -104.58 | 2.20s | Bonne |
| DQN | -97.20  | 70.32s | Excellente |

---

##  Configuration Commune

### Hyperparamètres Identiques
```python
alpha = 0.1      # Taux d'apprentissage (sauf DQN)
gamma = 0.95     # Facteur d'actualisation
epsilon = 0.2    # Exploration ε-greedy
max_steps = 100  # Étapes par épisode
```

### Politique d'Exploration
```python
# Epsilon-greedy pour tous les algorithmes
if np.random.uniform(0, 1) < epsilon:
    action = env.action_space.sample()  # Exploration
else:
    action = np.argmax(Q[state])        # Exploitation
```

---

##  Utilisation des Fichiers

### Exécution Individuelle
```bash
python on_policy.py          # SARSA seul
python off_policy.py         # Q-Learning seul
python td_learning.py        # Expected SARSA seul
python approximation_fonction.py  # DQN seul
```

### Comparaison Complète
```bash
python comparaison_finale.py     # Compare les 4 algorithmes
jupyter notebook analyse_comparative.ipynb  # Analyse détaillée
```

### Évaluation
```bash
python evaluation_politique.py   # Test qualité vs aléatoire
```

---

##  Fichiers de Sortie

- `comparaison_algorithmes.png` - Graphique de convergence
- `slide_*.png` - Graphiques pour présentation
- Tableaux de performance dans le terminal
- Notebook avec analyse complète

---

