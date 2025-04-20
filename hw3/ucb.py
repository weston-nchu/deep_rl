import numpy as np
import matplotlib.pyplot as plt

class UCB:
    def __init__(self, c, k):
        self.c = c
        self.k = k
        self.Q = np.zeros(k)
        self.N = np.zeros(k)
        self.t = 0

    def select_arm(self):
        self.t += 1
        ucb_values = self.Q + self.c * np.sqrt(np.log(self.t) / (self.N + 1e-5))
        return np.argmax(ucb_values)

    def update(self, arm, reward):
        self.N[arm] += 1
        self.Q[arm] += (reward - self.Q[arm]) / self.N[arm]

def simulate_bandit(ucb, steps):
    rewards = np.zeros(steps)
    optimal_action_counts = np.zeros(steps)
    optimal_action = np.argmax(ucb.Q)

    for step in range(steps):
        arm = ucb.select_arm()
        reward = np.random.normal(Q_true[arm], 1)  # Simulate reward with a normal distribution
        ucb.update(arm, reward)

        rewards[step] = reward
        if arm == optimal_action:
            optimal_action_counts[step] = 1

    cumulative_reward = np.cumsum(rewards)
    cumulative_optimal_action_counts = np.cumsum(optimal_action_counts) / np.arange(1, steps + 1)

    return cumulative_reward, cumulative_optimal_action_counts

# Simulation parameters
c = 2  # Exploration parameter
k = 10  # Number of arms
steps = 1000  # Number of steps to simulate

# True reward values for each arm (normally distributed)
Q_true = np.random.normal(0, 1, k)

# Initialize and simulate UCB bandit
ucb = UCB(c, k)
cumulative_reward, cumulative_optimal_action_counts = simulate_bandit(ucb, steps)

# Plot results
plt.figure(figsize=(12, 6))

# Plot cumulative reward
plt.subplot(1, 2, 1)
plt.plot(cumulative_reward, label='UCB')
plt.xlabel('Steps')
plt.ylabel('Cumulative Reward')
plt.title('UCB Bandit')
plt.legend()

# Plot optimal action percentage
plt.subplot(1, 2, 2)
plt.plot(cumulative_optimal_action_counts * 100, label='UCB')
plt.xlabel('Steps')
plt.ylabel('% Optimal Action')
plt.title('UCB Bandit')
plt.ylim(0, 100)
plt.legend()

plt.tight_layout()
plt.show()
