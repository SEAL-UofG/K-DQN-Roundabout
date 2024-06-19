import numpy as np
import matplotlib.pyplot as plt

def smooth_data(data, window_size=10):
    window = np.ones(window_size) / window_size
    smoothed_data = np.convolve(data, window, mode='same')#same




    return smoothed_data

def plot_with_confidence_interval(x, y, std, label, confidence=0.1):
    # 平滑处理均值和标准差
    y_smooth = smooth_data(y)
    std_smooth = smooth_data(std)

    plt.plot(x, y_smooth, label=label)
    plt.fill_between(x, y_smooth - confidence * std_smooth, y_smooth + confidence * std_smooth, alpha=0.2)

# 读取保存的数据
steps = np.load('steps.npy')
episodes = np.load('episodes.npy')
mean_rewards = np.load('mean_rewards.npy')
std_rewards = np.load('std_rewards.npy')
mean_speeds = np.load('mean_speeds.npy')
std_speeds = np.load('std_speeds.npy')
mean_collision_rates = np.load('mean_collision_rates.npy')
std_collision_rates = np.load('std_collision_rates.npy')

plt.figure(figsize=(12, 8))

# Rewards 曲线
plt.subplot(2, 2, 1)
plot_with_confidence_interval(episodes, mean_rewards, std_rewards, 'Rewards')
plt.xlabel('Episodes')
plt.xlim(-200, 11000)
plt.ylabel('Rewards')
plt.legend()

# Speeds 曲线
plt.subplot(2, 2, 2)
plot_with_confidence_interval(episodes, mean_speeds, std_speeds, 'Speeds')
plt.xlabel('Episodes')
plt.ylabel('Speeds')
plt.legend()

# Collision Rates 曲线
plt.subplot(2, 2, 3)
plot_with_confidence_interval(episodes, mean_collision_rates, std_collision_rates, 'Collision Rates')
plt.xlabel('Episodes')
plt.ylabel('Collision Rates')
plt.ylim(0, 1.0)  # 将y轴的范围限制在0到1之间，适用于碰撞率数据
plt.legend()

plt.tight_layout()
plt.show()
