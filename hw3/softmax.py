import numpy as np
import matplotlib.pyplot as plt

class SoftmaxBandit:
    def __init__(self, tau, k):
        self.tau = tau
        self.k = k
        self.Q = np.zeros(k)
        self.N = np.zeros(k)

    def select_arm(self):
        preferences = np.exp(self.Q / self.tau)
        probabilities = preferences / np.sum(preferences)
        return np.random.choice(self.k, p=probabilities)

    def update(self, arm, reward):
        self.N[arm] += 1
        self.Q[arm] += (reward - self.Q[arm]) / self.N[arm]

def simulate_bandit(softmax_bandit, steps):
    rewards = np.zeros(steps)
    optimal_action_counts = np.zeros(steps)
    optimal_action = np.argmax(Q_true)

    for step in range(steps):
        arm = softmax_bandit.select_arm()
        reward = np.random.normal(Q_true[arm], 1)
        softmax_bandit.update(arm, reward)

        rewards[step] = reward
        if arm == optimal_action:
            optimal_action_counts[step] = 1

    cumulative_reward = np.cumsum(rewards)
    cumulative_optimal_action_counts = np.cumsum(optimal_action_counts) / np.arange(1, steps + 1)

    return cumulative_reward, cumulative_optimal_action_counts

# Simulation parameters
tau = 0.1  # Temperature parameter
k = 10
steps = 1000

# True mean rewards
Q_true = np.random.normal(0, 1, k)

# Run Softmax simulation
softmax_bandit = SoftmaxBandit(tau, k)
cumulative_reward, cumulative_optimal_action_counts = simulate_bandit(softmax_bandit, steps)

# Plotting
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(cumulative_reward, label='Softmax')
plt.xlabel('Steps')
plt.ylabel('Cumulative Reward')
plt.title('Softmax Bandit')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(cumulative_optimal_action_counts * 100, label='Softmax')
plt.xlabel('Steps')
plt.ylabel('% Optimal Action')
plt.title('Softmax Bandit')
plt.ylim(0, 100)
plt.legend()

plt.tight_layout()
plt.show()