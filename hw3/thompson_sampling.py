import numpy as np
import matplotlib.pyplot as plt

class ThompsonSamplingBandit:
    def __init__(self, k):
        self.k = k
        self.alpha = np.ones(k)
        self.beta = np.ones(k)

    def select_arm(self):
        samples = np.random.beta(self.alpha, self.beta)
        return np.argmax(samples)

    def update(self, arm, reward):
        self.alpha[arm] += reward
        self.beta[arm] += 1 - reward

def simulate_bandit(ts_bandit, steps, true_probs):
    rewards = np.zeros(steps)
    optimal_action_counts = np.zeros(steps)
    optimal_action = np.argmax(true_probs)

    for step in range(steps):
        arm = ts_bandit.select_arm()
        reward = np.random.binomial(1, true_probs[arm])
        ts_bandit.update(arm, reward)

        rewards[step] = reward
        if arm == optimal_action:
            optimal_action_counts[step] = 1

    cumulative_reward = np.cumsum(rewards)
    cumulative_optimal_action_counts = np.cumsum(optimal_action_counts) / np.arange(1, steps + 1)

    return cumulative_reward, cumulative_optimal_action_counts

# Simulation parameters
k = 10
steps = 1000
true_probs = np.random.uniform(0.1, 0.9, k)

# Run Thompson Sampling simulation
ts_bandit = ThompsonSamplingBandit(k)
cumulative_reward, cumulative_optimal_action_counts = simulate_bandit(ts_bandit, steps, true_probs)

# Plotting
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(cumulative_reward, label='Thompson Sampling')
plt.xlabel('Steps')
plt.ylabel('Cumulative Reward')
plt.title('Thompson Sampling Bandit')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(cumulative_optimal_action_counts * 100, label='Thompson Sampling')
plt.xlabel('Steps')
plt.ylabel('% Optimal Action')
plt.title('Thompson Sampling Bandit')
plt.ylim(0, 100)
plt.legend()

plt.tight_layout()
plt.show()