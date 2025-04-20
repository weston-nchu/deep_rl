import numpy as np
import matplotlib.pyplot as plt

class EpsilonGreedy:
    def __init__(self, epsilon, k):
        self.epsilon = epsilon
        self.k = k
        self.Q = np.zeros(k)
        self.N = np.zeros(k)

    def select_arm(self):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.k)  # Explore: choose a random arm
        else:
            return np.argmax(self.Q)  # Exploit: choose the arm with the highest estimated value

    def update(self, arm, reward):
        self.N[arm] += 1
        self.Q[arm] += (reward - self.Q[arm]) / self.N[arm]

def simulate_bandit(epsilon, bandit, steps):
    rewards = np.zeros(steps)
    optimal_action_counts = np.zeros(steps)
    optimal_action = np.argmax(bandit.Q)

    for step in range(steps):
        arm = bandit.select_arm()
        reward = np.random.normal(Q_true[arm], 1)  # Simulate reward with a normal distribution
        bandit.update(arm, reward)

        rewards[step] = reward
        if arm == optimal_action:
            optimal_action_counts[step] = 1

    cumulative_reward = np.cumsum(rewards)
    cumulative_optimal_action_counts = np.cumsum(optimal_action_counts) / np.arange(1, steps + 1)

    return cumulative_reward, cumulative_optimal_action_counts

# Simulation parameters
epsilon = 0.1
k = 10  # Number of arms
steps = 1000  # Number of steps to simulate

# True reward values for each arm (normally distributed)
Q_true = np.random.normal(0, 1, k)

# Initialize and simulate Epsilon-Greedy bandit
bandit = EpsilonGreedy(epsilon, k)
cumulative_reward, cumulative_optimal_action_counts = simulate_bandit(epsilon, bandit, steps)

# Plot results
plt.figure(figsize=(12, 6))

# Plot cumulative reward
plt.subplot(1, 2, 1)
plt.plot(cumulative_reward, label='Epsilon-Greedy')
plt.xlabel('Steps')
plt.ylabel('Cumulative Reward')
plt.title('Epsilon-Greedy Bandit')
plt.legend()

# Plot optimal action percentage
plt.subplot(1, 2, 2)
plt.plot(cumulative_optimal_action_counts * 100, label='Epsilon-Greedy')
plt.xlabel('Steps')
plt.ylabel('% Optimal Action')
plt.title('Epsilon-Greedy Bandit')
plt.ylim(0, 100)
plt.legend()

plt.tight_layout()
plt.show()