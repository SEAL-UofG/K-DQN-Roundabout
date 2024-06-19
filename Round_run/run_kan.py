import argparse
import os
import time
import numpy as np
import torch
from kan import KAN
from torch.utils.tensorboard import SummaryWriter
from Round_run.buffer import ReplayBuffer

import sys

sys.path.append("../highway-env")
import gym


# mynote 为了gym的注册
from gym.envs.registration import register
from highway_env.envs import RoundaboutEnv
# mynote
# 这个导入语句确保了highway_env.py被执行，从而运行到register函数环境得以注册
import highway_env


def kan_train(net, target, data, optimizer, gamma=0.99, lamb=0.1, lamb_l1=0.1, lamb_entropy=0.1, lamb_coef=0.1, lamb_coefdiff=0.1):
    observations, actions, next_observations, rewards, terminations = data

    observations = observations.view(observations.shape[0], -1)
    next_observations = next_observations.view(next_observations.shape[0], -1)

    with torch.no_grad():
        next_q_values = net(next_observations)
        next_actions = next_q_values.argmax(dim=1)
        next_q_values_target = target(next_observations)
        target_max = next_q_values_target[range(len(next_q_values)), next_actions]
        td_target = rewards.flatten() + gamma * target_max * (1 - terminations.flatten().float())

    old_val = net(observations).gather(1, actions).squeeze()
    loss = torch.nn.functional.mse_loss(td_target, old_val)
    reg_ = 0.
    acts_scale = net.acts_scale
    for i in range(len(acts_scale)):
        vec = acts_scale[i].reshape(-1, )
        p = vec / torch.sum(vec)
        l1 = torch.sum(torch.abs(vec))
        entropy = - torch.sum(p * torch.log2(p + 1e-4))
        reg_ += lamb_l1 * l1 + lamb_entropy * entropy

    for i in range(len(net.act_fun)):
        coeff_l1 = torch.sum(torch.mean(torch.abs(net.act_fun[i].coef), dim=1))
        coeff_diff_l1 = torch.sum(torch.mean(torch.abs(torch.diff(net.act_fun[i].coef)), dim=1))
        reg_ += lamb_coef * coeff_l1 + lamb_coefdiff * coeff_diff_l1

    loss = loss + lamb * reg_
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate(net, env, n_episodes, run_name, episode):
    all_rewards = []
    all_speeds = []
    all_collision_rates = []

    for _ in range(n_episodes):
        episode_rewards = []
        episode_speeds = []


        state, _ = env.reset()
        done = False
        episode_reward = 0
        steps = 0

        while not done:
            # myrl render
            env.render()
            if isinstance(state, np.ndarray):
                state = torch.from_numpy(state).float()
            elif not isinstance(state, torch.Tensor):
                raise TypeError("State must be a numpy.ndarray or torch.Tensor")

            action = net(state).argmax(dim=1).item()
            next_state, reward, done, info = env.step((action,))

            state = next_state
            episode_reward += reward
            episode_speeds.append(np.mean([v.speed for v in env.controlled_vehicles]))
            steps += 1

        episode_rewards.append(episode_reward)
        all_rewards.append(episode_reward)
        all_speeds.append(episode_speeds)


    rewards_mean = np.mean(all_rewards)
    rewards_std = np.std(all_rewards)
    speeds_mean = np.mean(np.concatenate(all_speeds))
    speeds_std = np.std(np.concatenate(all_speeds))




    return rewards_mean, rewards_std, speeds_mean, speeds_std


def polyak_update(net, target_net, tau=0.01):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def main(args):
    eval_episode = 0
    # init env
    env = gym.make('round-multi-v2')
    # print(f"Observation Space: {env.observation_space}")
    # print(f"Action Space: {env.action_space}")
    env.config['seed'] = args.seed
    env.config['simulation_frequency'] = args.simulation_frequency
    env.config['duration'] = args.duration
    env.config['policy_frequency'] = args.policy_frequency
    env.config['COLLISION_REWARD'] = args.collision_reward
    env.config['HIGH_SPEED_REWARD'] = args.high_speed_reward
    env.config['HEADWAY_COST'] = args.headway_cost
    env.config['HEADWAY_TIME'] = args.headway_time
    env.config['MERGING_LANE_COST'] = args.merging_lane_cost
    env.config['traffic_density'] = args.traffic_density
    env.config['action_masking'] = args.action_masking
    env.config["reward_reshape"] = False
    env.config["action_inspector"] = False

    assert env.T % args.roll_out_n_steps == 0

    env_eval = gym.make('round-multi-v2')
    env_eval.config['seed'] = args.seed + 1
    env_eval.config['simulation_frequency'] = args.simulation_frequency
    env_eval.config['duration'] = args.duration
    env_eval.config['policy_frequency'] = args.policy_frequency
    env_eval.config['COLLISION_REWARD'] = args.collision_reward
    env_eval.config['HIGH_SPEED_REWARD'] = args.high_speed_reward
    env_eval.config['HEADWAY_COST'] = args.headway_cost
    env_eval.config['HEADWAY_TIME'] = args.headway_time
    env_eval.config['MERGING_LANE_COST'] = args.merging_lane_cost
    env_eval.config['traffic_density'] = args.traffic_density
    env_eval.config['action_masking'] = args.action_masking
    env_eval.config["reward_reshape"] = False
    env_eval.config["action_inspector"] = False
    env_eval.config["action"] = {
        "type": "MultiAgentAction",
        "action_config": {
            "type": "DiscreteMetaAction",
            "lateral": True,
            "longitudinal": True,
            "low_level": "PID"
        }}

    test_seeds = args.evaluation_seeds

    state_dim = 25  # 观测空间维度
    action_dim = 5  # 动作空间维度
    state_shape = (5, 5)  # Adjust to the shape of your observation space, which is (5, 5) matrix
    buffer = ReplayBuffer(args.replay_buffer_capacity, state_shape, action_dim)
    q_network = KAN(
        width=[state_dim, args.width, action_dim],
        grid=args.grid,
        k=3,
        bias_trainable=False,
        sp_trainable=False,
        sb_trainable=False,
    )
    target_network = KAN(
        width=[state_dim, args.width, action_dim],
        grid=args.grid,
        k=3,
        bias_trainable=False,
        sp_trainable=False,
        sb_trainable=False,
    )


    target_network.load_state_dict(q_network.state_dict())

    run_name = f"KAN_{int(time.time())}"



    optimizer = torch.optim.Adam(q_network.parameters(), args.learning_rate)

    best_eval_reward = -np.inf
    start_time = time.time()

    mean_rewards = []
    std_rewards = []
    mean_speeds = []
    std_speeds = []


    episode_rewards = []
    episode_speeds = []


    for episode in range(args.n_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        steps = 0

        while not done:
            # MYRL RENDER
            env.render()
            actions = []
            if episode < args.warm_up_episodes:
                actions = [env.action_space.sample()[0] for _ in env.controlled_vehicles]
            else:
                for vehicle_state in state:
                    state_tensor = torch.from_numpy(vehicle_state).double()
                    if state_tensor.dim() > 2:
                        state_tensor = state_tensor.squeeze()
                    if state_tensor.dim() == 1:
                        state_tensor = state_tensor.unsqueeze(0)

                    action = q_network(state_tensor).argmax(dim=1).item()
                    actions.append(action)

            next_state, reward, done, info = env.step(tuple(actions))
            buffer.add(torch.from_numpy(state), actions, torch.from_numpy(next_state), reward, done)
            state = next_state
            episode_reward += reward
            steps += 1


        if len(buffer) >= args.batch_size:
            for _ in range(args.train_steps):
                loss = kan_train(
                    q_network,
                    target_network,
                    buffer.sample(args.batch_size),
                    optimizer,
                    args.gamma,
                    lamb=0.01, lamb_l1=0.1, lamb_entropy=0.1, lamb_coef=0.1, lamb_coefdiff=0.1,
                )


            if episode % args.grid_update_freq == 0 and episode < args.stop_grid_update_step:
                state_batch = buffer.sample(args.batch_size)[0]
                state_batch = state_batch.view(state_batch.size(0), -1)
                q_network.update_grid_from_samples(state_batch)

            if episode % args.target_update_freq == 0:
                polyak_update(q_network, target_network, tau=0.01)
        # mychange
        if episode > args.prune_start_episode and episode % args.prune_freq == 0:
            q_network = q_network.prune(threshold=1e-2, mode="auto")

        if episode > args.symbolic_start_episode and episode % args.auto_symbolic_freq == 0:
            q_network.auto_symbolic(lib=['exp', 'sin', 'power'], verbose=1)

        episode_rewards.append(episode_reward)
        if len(env.controlled_vehicles) > 0:
            episode_speeds.append(np.mean([v.speed for v in env.controlled_vehicles]))
        else:
            episode_speeds.append(0)

        if episode % args.eval_freq == 0:
            rewards_mean, rewards_std, speeds_mean, speeds_std, = evaluate(
                q_network, env_eval, args.eval_episodes,run_name,episode)
            print(
                f"Episode: {episode}, Eval Reward: {rewards_mean:.2f} +/- {rewards_std:.2f}, Speed: {speeds_mean:.2f} +/- {speeds_std:.2f}")


            if rewards_mean > best_eval_reward:
                best_eval_reward = rewards_mean
                os.makedirs("models", exist_ok=True)
                torch.save(q_network.state_dict(), f"models/{run_name}_best.pth")


        if (episode + 1) % 200 == 0:
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"The code took {elapsed_time} seconds for 200 episodes.")
            start_time = time.time()
    os.makedirs("models", exist_ok=True)  # 创建 models 目录,如果它不存在的话
    torch.save(q_network.state_dict(), f"models/{run_name}_final.pth")
    torch.save(target_network.state_dict(), f"models/{run_name}_target_final.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KAN Agent")
    parser.add_argument("--n_episodes", type=int, default=20000, help="Number of episodes")
    parser.add_argument("--warm_up_episodes", type=int, default=200, help="Number of warm-up episodes")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--train_steps", type=int, default=5, help="Number of training steps per episode")
    parser.add_argument("--target_update_freq", type=int, default=100, help="Target network update frequency")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--replay_buffer_capacity", type=int, default=50000, help="Replay buffer capacity")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size")
    parser.add_argument("--width", type=int, default=64, help="Width of hidden layer")
    parser.add_argument("--grid", type=int, default=10, help="Grid size")
    parser.add_argument("--eval_freq", type=int, default=5, help="Evaluation frequency")
    parser.add_argument("--eval_episodes", type=int, default=5, help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=40, help="Random seed")
    parser.add_argument("--simulation_frequency", type=int, default=15, help="Simulation frequency")
    parser.add_argument("--duration", type=int, default=20, help="Duration")
    parser.add_argument("--policy_frequency", type=int, default=5, help="Policy frequency")
    parser.add_argument("--collision_reward", type=int, default=20, help="Collision reward")
    parser.add_argument("--high_speed_reward", type=int, default=4, help="High speed reward")
    parser.add_argument("--headway_cost", type=int, default=4, help="Headway cost")
    parser.add_argument("--headway_time", type=float, default=1.2, help="Headway time")
    parser.add_argument("--merging_lane_cost", type=int, default=4, help="Merging lane cost")
    parser.add_argument("--traffic_density", type=int, default=1, help="Traffic density")
    parser.add_argument("--action_masking", type=bool, default=True, help="Action masking")
    parser.add_argument("--roll_out_n_steps", type=int, default=1, help="Roll-out steps")
    parser.add_argument("--evaluation_seeds", type=str, default=','.join([str(i) for i in range(0, 600, 20)]),
                        help="Evaluation seeds")
    parser.add_argument("--grid_update_freq", type=int, default=100, help="Grid update frequency")
    parser.add_argument("--stop_grid_update_step", type=int, default=5000, help="Stop grid update at this step")
    parser.add_argument("--prune_start_episode", type=int, default=10000, help="Start pruning from this episode")
    parser.add_argument("--prune_freq", type=int, default=2000, help="Pruning frequency")
    parser.add_argument("--symbolic_start_episode", type=int, default=15000,
                        help="Start auto-symbolic from this episode")
    parser.add_argument("--auto_symbolic_freq", type=int, default=5000, help="Auto-symbolic frequency")

    args = parser.parse_args()

    main(args)