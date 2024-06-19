import torch
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity, observation_shape, action_space):
        self.capacity = capacity
        self.observations = torch.zeros((capacity, *observation_shape))  # 调整为二维观测数据的形状
        self.actions = torch.zeros((capacity, 1), dtype=torch.long)  # 存储离散动作
        self.next_observations = torch.zeros((capacity, *observation_shape))  # 调整为二维观测数据的形状
        self.rewards = torch.zeros(capacity, 1)
        self.terminations = torch.zeros(capacity, 1, dtype=torch.bool)  # 使用布尔类型存储终止状态
        self.cursor = 0

    def add(self, observation, action, next_observation, reward, termination):
        index = self.cursor % self.capacity

        # Ensure input is a torch.Tensor
        if not isinstance(observation, torch.Tensor):
            observation = torch.tensor(observation, dtype=torch.float32)
        if not isinstance(next_observation, torch.Tensor):
            next_observation = torch.tensor(next_observation, dtype=torch.float32)

        # Check dimensions and reshape if necessary
        expected_shape = (5, 5)  # Assuming this is the correct shape
        if observation.shape != expected_shape:
            observation = observation.view(expected_shape)
        if next_observation.shape != expected_shape:
            next_observation = next_observation.view(expected_shape)

        # Direct assignment after confirming dimensions
        self.observations[index] = observation.clone().detach()
        self.actions[index] = torch.tensor([action], dtype=torch.long)
        self.next_observations[index] = next_observation.clone().detach()
        self.rewards[index] = torch.tensor([reward], dtype=torch.float32)
        self.terminations[index] = torch.tensor([termination], dtype=torch.bool)

        self.cursor += 1

    def sample(self, batch_size):
        idx = np.random.choice(min(self.cursor, self.capacity), size=batch_size, replace=False)
        return (
            self.observations[idx],
            self.actions[idx],
            self.next_observations[idx],
            self.rewards[idx],
            self.terminations[idx]
        )

    def __len__(self):
        return min(self.cursor, self.capacity)

# Example of initializing ReplayBuffer
# observation_shape = (5, 5) because the observation is a 5x5 matrix
# action_space = 5 since there are 5 discrete actions
state_shape = (5, 5)  # Since each observation is a 5x5 matrix
buffer = ReplayBuffer(capacity=1000, observation_shape=(5, 5), action_space=5)
