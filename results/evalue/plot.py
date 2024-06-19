# import numpy as np
# import matplotlib.pyplot as plt
#
# def smooth_data(data, window_size=1):
#     window = np.ones(window_size) / window_size
#     smoothed_data = np.convolve(data, window, mode='same')
#     return smoothed_data
#
# def plot_with_confidence_interval(x, y, std, label, confidence=0.1):
#     y_smooth = smooth_data(y)
#     std_smooth = smooth_data(std)
#     plt.plot(x, y_smooth, label=label)
#     plt.fill_between(x, y_smooth - confidence * std_smooth, y_smooth + confidence * std_smooth, alpha=0.2)
#
# # 加载新的评估数据
# eval_episodes = np.load('/home/h/Paper_Round/no_reshape/Round_MARL/results/evalue/eval_episodes.npy')
# eval_rewards_mean = np.load('/home/h/Paper_Round/no_reshape/Round_MARL/results/evalue/eval_rewards_mean.npy')
# eval_rewards_std = np.load('/home/h/Paper_Round/no_reshape/Round_MARL/results/evalue/eval_rewards_std.npy')
# eval_speeds_mean = np.load('/home/h/Paper_Round/no_reshape/Round_MARL/results/evalue/eval_speeds_mean.npy')
# eval_speeds_std = np.load('/home/h/Paper_Round/no_reshape/Round_MARL/results/evalue/eval_speeds_std.npy')
# eval_collision_rates_mean = np.load('/home/h/Paper_Round/no_reshape/Round_MARL/results/evalue/eval_collision_rates_mean.npy')
# eval_collision_rates_std = np.load('/home/h/Paper_Round/no_reshape/Round_MARL/results/evalue/eval_collision_rates_std.npy')
#
# plt.figure(figsize=(12, 8))
#
# # Rewards 曲线
# plt.subplot(2, 2, 1)
# plot_with_confidence_interval(eval_episodes, eval_rewards_mean, eval_rewards_std, 'Evaluation Rewards')
# plt.xlabel('Evaluation Episodes')
# plt.ylabel('Rewards')
# plt.legend()
#
# # Speeds 曲线
# plt.subplot(2, 2, 2)
# plot_with_confidence_interval(eval_episodes, eval_speeds_mean, eval_speeds_std, 'Evaluation Speeds')
# plt.xlabel('Evaluation Episodes')
# plt.ylabel('Speeds')
# plt.legend()
#
# # Collision Rates 曲线
# plt.subplot(2, 2, 3)
# plot_with_confidence_interval(eval_episodes, eval_collision_rates_mean, eval_collision_rates_std, 'Evaluation Collision Rates')
# plt.xlabel('Evaluation Episodes')
# plt.ylabel('Collision Rates')
# plt.ylim(0, 1.0)  # 将y轴的范围限制在0到1之间，适用于碰撞率数据
# plt.legend()
#
# plt.tight_layout()
# plt.show()


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
# 加载新的评估数据
eval_episodes = np.load('/home/h/Paper_Round/no_reshape/Round_MARL/results/evalue/eval_episodes.npy')
eval_rewards_mean = np.load('/home/h/Paper_Round/no_reshape/Round_MARL/results/evalue/eval_rewards_mean.npy')
eval_rewards_std = np.load('/home/h/Paper_Round/no_reshape/Round_MARL/results/evalue/eval_rewards_std.npy')
eval_speeds_mean = np.load('/home/h/Paper_Round/no_reshape/Round_MARL/results/evalue/eval_speeds_mean.npy')
eval_speeds_std = np.load('/home/h/Paper_Round/no_reshape/Round_MARL/results/evalue/eval_speeds_std.npy')
eval_collision_rates_mean = np.load('/home/h/Paper_Round/no_reshape/Round_MARL/results/evalue/eval_collision_rates_mean.npy')
eval_collision_rates_std = np.load('/home/h/Paper_Round/no_reshape/Round_MARL/results/evalue/eval_collision_rates_std.npy')

plt.figure(figsize=(12, 8))

# Rewards 曲线
plt.subplot(2, 2, 1)
plot_with_confidence_interval(eval_episodes, eval_rewards_mean, eval_rewards_std, 'Evaluation Rewards')
plt.xlabel('Evaluation Episodes')
plt.ylabel('Rewards')
plt.legend()

# Speeds 曲线
plt.subplot(2, 2, 2)
plot_with_confidence_interval(eval_episodes, eval_speeds_mean, eval_speeds_std, 'Evaluation Speeds')
plt.xlabel('Evaluation Episodes')
plt.ylabel('Speeds')
plt.legend()

# Collision Rates 曲线
plt.subplot(2, 2, 3)
plot_with_confidence_interval(eval_episodes, eval_collision_rates_mean, eval_collision_rates_std, 'Evaluation Collision Rates')
plt.xlabel('Evaluation Episodes')
plt.ylabel('Collision Rates')
plt.ylim(0, 1.0)  # 将y轴的范围限制在0到1之间，适用于碰撞率数据
plt.legend()

plt.tight_layout()
plt.show()
