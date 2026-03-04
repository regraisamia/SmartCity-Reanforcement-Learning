# 🚦 Smart City Traffic Control - Reinforcement Learning

A comprehensive comparison of **4 Reinforcement Learning algorithms** applied to urban traffic light optimization using **Gymnasium**. This project implements and benchmarks SARSA, Q-Learning, Expected SARSA, and Deep Q-Network (DQN) to minimize traffic congestion at a simulated intersection.

---

## 📑 Table of Contents
- [Project Architecture](#project-architecture)
- [Implemented Algorithms](#implemented-algorithms)
- [Environment Specifications](#environment-specifications)
- [Comparison Metrics](#comparison-metrics)
- [Usage](#usage)
- [Results](#results)

---

## 🏗️ Project Architecture

### Main Files

| File | Type | Description |
|------|------|-------------|
| `env_smartcity.py` | Environment | Urban intersection with Gymnasium |
| `on_policy.py` | Algorithm | SARSA (On-Policy) |
| `off_policy.py` | Algorithm | Q-Learning (Off-Policy) |
| `td_learning.py` | Algorithm | Expected SARSA (TD Learning) |
| `approximation_fonction.py` | Algorithm | DQN (Function Approximation) |
| `comparaison_finale.py` | Analysis | Comparison of all 4 algorithms |
| `evaluation_politique.py` | Evaluation | Policy quality testing |
| `analyse_comparative.ipynb` | Notebook | Detailed results with visualizations |

---

## 🤖 Implemented Algorithms

### 1. SARSA - On-Policy (`on_policy.py`)

**Principle:** Follows the current policy during learning

**Update Rule:**
```
Q(s,a) ← Q(s,a) + α[R + γQ(s',a') - Q(s,a)]
```

**Parameters:**
- `alpha = 0.1` - Learning rate
- `gamma = 0.95` - Discount factor
- `epsilon = 0.2` - Exploration rate
- `episodes = 500` - Number of episodes

**Characteristics:**
- Conservative (follows its own policy)
- Stable but may be suboptimal

---

### 2. Q-Learning - Off-Policy (`off_policy.py`)

**Principle:** Learns the optimal policy independently of the followed policy

**Update Rule:**
```
Q(s,a) ← Q(s,a) + α[R + γ max Q(s',a) - Q(s,a)]
```

**Parameters:**
- `alpha = 0.1` - Learning rate
- `gamma = 0.95` - Discount factor
- `epsilon = 0.2` - Exploration rate
- `episodes = 500` - Number of episodes

**Characteristics:**
- Optimistic (aims for the best action)
- Converges to optimal policy

---

### 3. Expected SARSA - TD Learning (`td_learning.py`)

**Principle:** Computes the expectation of Q-values instead of the maximum

**Update Rule:**
```
Q(s,a) ← Q(s,a) + α[R + γ E[Q(s',a')] - Q(s,a)]
```

**Expectation Calculation:**
```python
expected_q = Σ π(a'|s') × Q(s',a')
```

**Parameters:**
- `alpha = 0.1` - Learning rate
- `gamma = 0.95` - Discount factor
- `epsilon = 0.2` - Exploration rate
- `episodes = 500` - Number of episodes

**Characteristics:**
- Combines advantages of on-policy and off-policy
- More stable than Q-Learning

---

### 4. DQN - Function Approximation (`approximation_fonction.py`)

**Principle:** Uses a neural network to approximate Q(s,a)

**Network Architecture:**
```
Input(1) → Dense(64) → ReLU → Dense(64) → ReLU → Output(4)
```

**Parameters:**
- `lr = 0.001` - Learning rate (Adam optimizer)
- `gamma = 0.95` - Discount factor
- `epsilon = 0.2` - Exploration rate
- `episodes = 200` - Fewer episodes (slower training)

**Characteristics:**
- Handles large state spaces
- Slower but more expressive

---

## 🌆 Environment Specifications (`env_smartcity.py`)

### Gymnasium Configuration

```python
observation_space = spaces.Discrete(100)  # 100 distinct states
action_space = spaces.Discrete(4)         # 4 possible actions
```

### States (100 possible)
- **Encoding:** `state = (cars_NS × 10) + cars_EO`
- **Example:** State 45 = 4 cars North-South + 5 cars East-West
- **Range:** 0-99 (0-9 cars per axis)

### Actions (4 possible)
- **Action 0:** GREEN light North-South
- **Action 1:** GREEN light East-West
- **Action 2:** YELLOW light North-South
- **Action 3:** YELLOW light East-West

### System Dynamics
```python
# GREEN light → Evacuates 2 cars, 1 new car arrives on the other axis
if action == 0:  # NS Green
    cars_ns = max(0, cars_ns - 2)
    cars_eo = min(9, cars_eo + 1)
```

### Reward Function
```python
reward = -(cars_ns + cars_eo)
```
**Rationale:** Penalizes total congestion. Closer to 0 = better performance.

---

## 📊 Comparison Metrics

### Evaluated Metrics
1. **Execution Time** - Training speed
2. **Final Performance** - Average reward (last 50 episodes)
3. **Stability** - Standard deviation of performance
4. **Convergence** - Speed of improvement
5. **Optimal Policy** - Quality of learned strategy

### Typical Results
| Algorithm | Performance | Time | Stability |
|-----------|-------------|------|-----------|
| SARSA | -103.98 | 0.69s | Medium |
| Q-Learning | -105.44 | 0.97s | Low |
| Expected SARSA | -104.58 | 2.20s | Good |
| DQN | -97.20  | 70.32s | Excellent |

---

## ⚙️ Common Configuration

### Shared Hyperparameters
```python
alpha = 0.1      # Learning rate (except DQN)
gamma = 0.95     # Discount factor
epsilon = 0.2    # ε-greedy exploration
max_steps = 100  # Steps per episode
```

### Exploration Policy
```python
# Epsilon-greedy for all algorithms
if np.random.uniform(0, 1) < epsilon:
    action = env.action_space.sample()  # Exploration
else:
    action = np.argmax(Q[state])        # Exploitation
```

---

## 🚀 Usage

### Individual Execution
```bash
python on_policy.py                  # SARSA only
python off_policy.py                 # Q-Learning only
python td_learning.py                # Expected SARSA only
python approximation_fonction.py     # DQN only
```

### Complete Comparison
```bash
python comparaison_finale.py         # Compare all 4 algorithms
jupyter notebook analyse_comparative.ipynb  # Detailed analysis
```

### Evaluation
```bash
python evaluation_politique.py       # Test quality vs random policy
```

---

## 📈 Results

### Output Files
- `comparaison_algorithmes.png` - Convergence graph
- `slide_*.png` - Presentation graphics
- Performance tables in terminal
- Notebook with complete analysis

### Key Findings
- **DQN** achieves the best performance but requires significantly more training time
- **SARSA** offers the fastest training with reasonable performance
- **Expected SARSA** provides the best stability-performance tradeoff
- **Q-Learning** is optimistic but less stable than Expected SARSA

---

## 💻 Technologies Used

- **Reinforcement Learning:** Gymnasium, NumPy
- **Deep Learning:** TensorFlow/Keras (for DQN)
- **Visualization:** Matplotlib, Seaborn
- **Analysis:** Jupyter Notebook, Pandas

---

## 🎓 About

This project demonstrates the practical application of various Reinforcement Learning algorithms to a real-world urban traffic optimization problem. It provides a comprehensive comparison framework for understanding the tradeoffs between different RL approaches.

---

## 👤 Author

**Samia Regrai**

