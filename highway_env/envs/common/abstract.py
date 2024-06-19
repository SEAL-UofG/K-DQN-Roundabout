
# 用于创建车辆驾驶在道路上的各种任务的通用仿真环境

import copy
import os
from typing import List, Tuple, Optional, Callable
import gym
import random
from gym import Wrapper
import numpy as np
from queue import PriorityQueue

# from highway_env import utils
from ... import utils

from highway_env.envs.common.action import action_factory, Action, DiscreteMetaAction, ActionType
from highway_env.envs.common.observation import observation_factory, ObservationType
from highway_env.envs.common.finite_mdp import finite_mdp
from highway_env.envs.common.graphics import EnvViewer
from highway_env.vehicle.behavior import IDMVehicle, LinearVehicle
from highway_env.vehicle.controller import MDPVehicle
from highway_env.vehicle.kinematics import Vehicle
from highway_env.envs.common.idm_controller import idm_controller, generate_actions
from highway_env.envs.common.mdp_controller import mdp_controller
# from highway_env.road.objects import Obstacle, Landmark

import math

Observation = np.ndarray
DEFAULT_WIDTH: float = 4  # width of the straight lane

# 名为 AbstractEnv 的类，它继承自 gym.Env 类
class AbstractEnv(gym.Env):
    """
    A generic environment for various tasks involving a vehicle driving on a road.

    The environment contains a road populated with vehicles, and a controlled ego-vehicle that can change lane and
    speed. The action space is fixed, but the observation space and reward function must be defined in the
    environment implementations.
    """
    # 定义了观察和动作的类型。
    observation_type: ObservationType
    action_type: ActionType
    # 可选的自动渲染回调函数。
    automatic_rendering_callback: Optional[Callable]
    # 包含渲染模式的元数据
    metadata = {'render.modes': ['human', 'rgb_array']}
    # 定义了观察中可以感知到的最远距离
    PERCEPTION_DISTANCE = 6.0 * MDPVehicle.MAX_SPEED
    """The maximum distance of any vehicle present in the observation [m]"""

    def __init__(self, config: dict = None) -> None:
        # Configuration
        self.config = self.default_config()
        if config:
            self.config.update(config)

        # Seeding
        # mynote 修改这里
        self.np_random = None
        self.seed = self.config["seed"]

        # Scene
        self.road = None
        self.controlled_vehicles = []

        # Spaces
        self.action_type = None
        self.action_space = None
        self.observation_type = None
        self.observation_space = None
        self.define_spaces()

        # Running
        # 跟踪仿真时间、执行的步骤数、环境是否结束以及仿真的总持续时间。
        self.time = 0  # Simulation time
        self.steps = 0  # Actions performed
        self.done = False
        # myrl 多少步
        self.T = int(self.config["duration"] * self.config["policy_frequency"])

        # Rendering
        # 这些属性和方法用于管理环境的渲染和视图更新。
        self.viewer = None
        self.automatic_rendering_callback = None
        self.should_update_rendering = True
        self.rendering_mode = 'human'
        self.enable_auto_render = False

        # 义了环境的一些特定于任务的属性，例如可能的结束状态、动作是否安全、所有可能的动作。
        # myrl 新增定义属性
        self.ends = [220, 100, 100, 100]  # Before, converging, merge, after
        self.action_is_safe = True
        # 当前车辆可做的动作列表
        self.ACTIONS_ALL = {'LANE_LEFT': 0, # 左拐
                            'IDLE': 1, # 空闲
                            'LANE_RIGHT': 2, #右拐
                            'FASTER': 3, # 加速
                            'SLOWER': 4} # 减速

        self.reset()

    @property
    # 可以获取和设置当前控制车辆的属性
    def vehicle(self) -> Vehicle:
        """First (default) controlled vehicle."""
        return self.controlled_vehicles[0] if self.controlled_vehicles else None

    @vehicle.setter
    def vehicle(self, vehicle: Vehicle) -> None:
        """Set a unique controlled vehicle."""
        self.controlled_vehicles = [vehicle]

    @classmethod
    # 该方法返回一个代表默认环境配置的字典
    # 它不需要实例化对象即可被类本身调用。cls 是指向类本身的引用。这个方法返回一个字典
    def default_config(cls) -> dict:
        """
        Default environment configuration.

        Can be overloaded in environment implementations, or by calling configure().
        :return: a configuration dict
        """
        return {
            # 分别定义了环境中观察和行动的类型
            "observation": {
                "type": "TimeToCollision"
            },
            "action": {
                "type": "DiscreteMetaAction"
            },
            # simulation_frequency 和 policy_frequency 定义了模拟和策略执行的频率。
            "simulation_frequency": 15,  # [Hz]
            "policy_frequency": 5,  # [Hz]
            # 指定了其他车辆的类型。
            "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
            # 配置了渲染窗口的尺寸和缩放比例。
            "screen_width": 900,
            "screen_height": 900,
            "centering_position": [0.3, 0.5],
            "scaling": 4,
            # 等选项控制了渲染和安全相关的配置。
            "show_trajectories": False,
            "render_agent": True,
            "action_inspector": False,
            "offscreen_rendering": os.environ.get("OFFSCREEN_RENDERING", "0") == "1",
            "manual_control": False,
            # myrl 新增
            "real_time_rendering": False,
            "n_step": 3,  # do n step prediction
            "seed": 0,
            "action_masking": True
        }
    # 法用于设置随机数生成器的种子，以便于实验的可重复
    def seed(self, seeding: int = None) -> List[int]:
        seed = np.random.seed(self.seed)
        return [seed]
    # configure 方法接收一个字典作为参数，并用它来更新实例的配置
    def configure(self, config: dict) -> None:
        if config:
            self.config.update(config)

    # 方法根据配置来设定观察和行动的类型及其空间
    def define_spaces(self) -> None:
        """
        Set the types and spaces of observation and action from config.
        """
        self.observation_type = observation_factory(self, self.config["observation"])
        self.action_type = action_factory(self, self.config["action"])
        self.observation_space = self.observation_type.space()
        self.action_space = self.action_type.space()

    def _reward(self, action: Action) -> float:
        """
        Return the reward associated with performing a given action and ending up in the current state.

        :param action: the last action performed
        :return: the reward
        """
        raise NotImplementedError

    def _is_terminal(self) -> bool:
        """
        Check whether the current state is a terminal state

        :return:is the state terminal
        """
        raise NotImplementedError

    def _cost(self, action: Action) -> float:
        """
        A constraint metric, for budgeted MDP.

        If a constraint is defined, it must be used with an alternate reward that doesn't contain it as a penalty.
        :param action: the last action performed
        :return: the constraint signal, the alternate (constraint-free) reward
        """
        raise NotImplementedError

    #接受三个参数：is_training 表示是否处于训练模式，默认为 True；testing_seeds 用于非训练模式时的随机种子
    # ，以保证测试结果的一致性，默认为 0；num_CAV 表示控制车辆的数量，默认为 0
    # 返回一个观测结果类型
    # myrl 修改了这个函数的方法
    def reset(self, is_training=True, testing_seeds=0, num_CAV=0) -> Observation:
        """
        Reset the environment to it's initial configuration

        :return: the observation of the reset state
        """
        # 判断是否处于训练模式
        # 这段代码通过在不同模式下使用确定性的种子值来初始化随机数生成器，
        # 保证了实验在相同条件下的可重复性。在训练模式下，
        # 使用内部种子确保了每次实验的一致性；而在测试模式下，
        # 通过使用外部传入的固定种子，无论实验进行多少次，
        # 都能保证测试环境的一致性和结果的可比较性。
        # 这种方法是科学实验、机器学习模型训练和评估中非常重要的实践，
        # 它确保了实验结果的准确性和信度。
        if is_training:
            # 如果处于训练模式，这两行代码使用环境内部保存的种子（self.seed）
            # 初始化 NumPy 和 Python 的随机数生成器，以确保实验可重复性。
            np.random.seed(self.seed)
            random.seed(self.seed)
        else:
            # 使用传入的 testing_seeds 初始化随机数生成器，以确保测试的一致性。
            np.random.seed(testing_seeds)
            random.seed(testing_seeds)
        # 调用 define_spaces 方法，首先根据动作空间设置控制车辆类。
        self.define_spaces()  # First, to set the controlled vehicle class depending on action space
        # 重置时间和步数计数器
        self.time = self.steps = 0
        # 更新环境的随机种子，为下一次重置或下一个实验准备
        self.seed += 1
        # 将环境状态设置为非终止。
        self.done = False
        # 清空车辆速度和位置列表。
        self.vehicle_speed = []
        self.vehicle_pos = []
        # 根据输入参数进行场景创建和初始化的实际执行部分
        self._reset(num_CAV=num_CAV)
        self.define_spaces()  # Second, to link the obs and actions to the vehicles once the scene is created
        # set the vehicle id for visualizing
        # 循环为道路上的每辆车分配一个 ID
        for i, v in enumerate(self.road.vehicles):
            v.id = i
        #     ：调用观察类型的 observe 方法，收集当前环境状态的观察结果。
        obs = self.observation_type.observe()
        # get action masks
        # mynote 这里是设定动作掩码（action masks）的地方，与你之前询问的动作掩码概念直接相关。
        #  它根据环境中可行的动作动态创建动作掩码，这些掩码将用于后续的决策过程中，以确保智能体仅考虑那些在当前状态下可行的动作
        # 如果配置中启用了动作掩码，
        if self.config["action_masking"]:
            # 根据环境中控制的车辆数量创建一个初始全为0的动作掩码列表 available_actions
            # _get_available_actions 方法获取每辆控制车辆的可行动作，并在相应的动作掩码列表中将这些动作标记为1，表示可行
            # print("controlled_vehicles===",len(self.controlled_vehicles))
            available_actions = [[0] * self.n_a] * len(self.controlled_vehicles)
            for i in range(len(self.controlled_vehicles)):
                available_action = self._get_available_actions(self.controlled_vehicles[i], self)
                # print("action list==",available_action)
                for a in available_action:
                    available_actions[i][a] = 1
        else:
            # 为所有控制车辆创建一个初始全为1的动作掩码列表，意味着所有动作都被视为可行。
            available_actions = [[1] * self.n_a] * len(self.controlled_vehicles)
        # 返回处理后的观察结果和动作掩码列表
        return np.asarray(obs).reshape((len(obs), -1)), np.array(available_actions)

    # 这个函数在派生类的mergev1中重写
    def _reset(self, num_CAV=1) -> None:
        """
        Reset the scene: roads and vehicles.

        This method must be overloaded by the environments.
        """
        raise NotImplementedError()


    def _get_available_actions(self, vehicle, env_copy):
        """
        Get the list of currently available actions.
        Lane changes are not available on the boundary of the road, and speed changes are not available at
        maximal or minimal speed.
        :return: the list of available actions
        """
        # if not isinstance(self.action_type, DiscreteMetaAction):
        #     raise ValueError("Only discrete meta-actions can be unavailable.")
        actions = [env_copy.ACTIONS_ALL['IDLE']]
        for l_index in env_copy.road.network.side_lanes(vehicle.lane_index):
            if l_index[2] < vehicle.lane_index[2] \
                    and env_copy.road.network.get_lane(l_index).is_reachable_from(vehicle.position):
                actions.append(env_copy.ACTIONS_ALL['LANE_LEFT'])
            if l_index[2] > vehicle.lane_index[2] \
                    and env_copy.road.network.get_lane(l_index).is_reachable_from(vehicle.position):
                actions.append(env_copy.ACTIONS_ALL['LANE_RIGHT'])
        if vehicle.speed_index <= 2:
            actions.append(env_copy.ACTIONS_ALL['FASTER'])
        if vehicle.speed_index >= 3:
            actions.append(env_copy.ACTIONS_ALL['SLOWER'])
        return actions



    def _get_available_actions1(self, action, vehicle, env_copy):
        roundabout_lanes = [
            ("se", "se1", 0), ("se1", "ex1", 0), ("ex1", "ex", 0), ("ex", "ee", 0), ("ee", "ee1", 0),
            ("ee1", "nx1", 0), ("nx1", "nx", 0), ("nx", "ne", 0), ("ne", "ne1", 0), ("ne1", "wx1", 0),
            ("wx1", "wx", 0), ("wx", "we", 0), ("we", "we1", 0), ("we1", "sx1", 0), ("sx1", "sx", 0),
            ("sx", "se", 0), ("se", "se1", 1), ("se1", "ex1", 1), ("ex1", "ex", 1), ("ex", "ee", 1),
            ("ee", "ee1", 1), ("ee1", "nx1", 1), ("nx1", "nx", 1), ("nx", "ne", 1), ("ne", "ne1", 1),
            ("ne1", "wx1", 1), ("wx1", "wx", 1), ("wx", "we", 1), ("we", "we1", 1), ("we1", "sx1", 1),
            ("sx1", "sx", 1), ("sx", "se", 1)
        ]

        v_fr, v_rr, v_fl, v_rl = env_copy.road.surrounding_vehicles(vehicle)
        # print ("available_actions1===",action)
        if vehicle.lane_index in roundabout_lanes:
            # 在内道 且 车想右拐
            if vehicle.lane_index[2] == 0 and action == 2:
                # 前面有车,且安全间隙大,且左边车道没车
                if v_fr is not None and 20 < np.linalg.norm(
                        v_fr.position - vehicle.position) and v_rl is None and v_fl is None:
                    return env_copy.ACTIONS_ALL['LANE_RIGHT']
                # 前面有车,且安全间隙小,或者左右有车
                elif (v_fr is not None and 20 >= np.linalg.norm(
                        v_fr.position - vehicle.position)) or v_rl is not None or v_fl is not None:
                    return env_copy.ACTIONS_ALL['SLOWER']

            # 在外道 且 车想左拐
            elif vehicle.lane_index[2] == 0 and  action == 0:
                # if v_fl is None or v_rl is None:
                #     return env_copy.ACTIONS_ALL['LANE_RIGHT']

                # 前面有车,且安全间隙大,且左边车道没车
                if v_fr is not None and 20 < np.linalg.norm(
                        v_fr.position - vehicle.position) and v_rl is None and v_fl is None:
                    return env_copy.ACTIONS_ALL['LANE_LEFT']
                # 前面有车,且安全间隙小,或者左右有车
                elif (v_fr is not None and 20 >= np.linalg.norm(
                        v_fr.position - vehicle.position)) or v_rl is not None or v_fl is not None:
                    return env_copy.ACTIONS_ALL['SLOWER']
                # 前面有车,且安全间隙小,或者左右有车
                # elif (v_fr is not None and 20 >= np.linalg.norm(
                #     v_fr.position - vehicle.position)) or v_rl is not None or v_fl is not None:
                #     return env_copy.ACTIONS_ALL['SLOWER']

            elif vehicle.lane_index[2] == 0 or vehicle.lane_index[2] == 1 and action != 0 and action != 2:
                if vehicle.speed_index <= 2:
                    return env_copy.ACTIONS_ALL['FASTER']
                if vehicle.speed_index >= 3:
                    return env_copy.ACTIONS_ALL['SLOWER']

        # 在汇入口
        elif (vehicle.lane_index == ("ses", "se", 0) or vehicle.lane_index == ("ses1", "se1", 0) or
              vehicle.lane_index == ("ees", "ee", 0) or vehicle.lane_index == ("ees1", "ee1", 0) or
              vehicle.lane_index == ("nes", "ne", 0) or vehicle.lane_index == ("nes1", "ne1", 0) or
              vehicle.lane_index == ("wes", "we", 0) or vehicle.lane_index == ("wes1", "we1", 0)):
            # 前面有车,且安全间隙小
            if v_fr is not None and 20 >= np.linalg.norm(v_fr.position - vehicle.position):
                return env_copy.ACTIONS_ALL['SLOWER']
            else:
                if vehicle.speed_index <= 2:
                    return env_copy.ACTIONS_ALL['FASTER']
                if vehicle.speed_index >= 3:
                    return env_copy.ACTIONS_ALL['SLOWER']

        return env_copy.ACTIONS_ALL['IDLE']






    def check_safety_room(self, vehicle, action, surrounding_vehicles, env_copy, time_steps):
        """
        para: vehicle: the ego vehicle
              surrounding_vehicles: [v_fl, v_rl, v_fr, v_rr]
              env_copy: copy of self
              vehicle.trajectories = [vehicle.position, vehicle.heading, vehicle.speed]
              return: the minimum safety room with surrounding vehicles in the trajectory
        """


        return 0



    def virtual_simulate(self, action: Optional[Action] = None) -> None:
        """虚拟的执行一步，看看结果如何."""
        frames = int(self.config["simulation_frequency"] // self.config["policy_frequency"])
        self.virtual_road = copy.deepcopy(self.road)  # 深拷贝能够隔离原有的一切

        if action is not None:
            virtual_actions = {
                'LANE_LEFT': 0,
                'IDLE': 1,
                'LANE_RIGHT': 2,
                'FASTER': 3,
                'SLOWER': 4
            }
            # print("action:", action)  # 输出 action 字典，例如 {1: 1}
            # print(self.virtual_road.controlled_vehicles)

            for index, vehicle in enumerate(self.virtual_road.controlled_vehicles):
                if index in action:  # 确保 action 字典中有当前车辆的索引
                    vehicle_action = list(virtual_actions.keys())[action[index]]  # 获取动作字符串
                    vehicle.act(vehicle_action)
                    # print(vehicle_action)

        for frame in range(frames):
            self.virtual_road.act()
            self.virtual_road.step(1 / self.config["simulation_frequency"])

        # 虚拟步数
    def virtual_step(self, action: Action) -> Tuple[float, dict]:
        """
        虚拟的step，不会真的更新道路状态
        """
        pass

    def virtual_reward(self, action: int) -> float:
        # 返回一个虚拟的奖励，
        raise NotImplementedError()

    def virtual_info(self, action: int) -> dict:
        # 返回一个虚拟的信息
        raise NotImplementedError()

    def action_inspector(self, actions):
        """"
        implementation of action inspector
        """

        return tuple(actions)





    def step(self, action: Action) -> Tuple[Observation, float, bool, dict]:
        """
        Perform an action and step the environment dynamics.

        The action is executed by the ego-vehicle, and all other vehicles on the road performs their default behaviour
        for several simulation timesteps until the next decision making step.

        :param action: the action performed by the ego-vehicle
        :return: a tuple (observation, reward, terminal, info)
        """
        # 始化平均速度为0
        average_speed = 0
        # 检查道路和车辆是否已经被正确初始化，如果没有，则抛出异常。
        if self.road is None or self.vehicle is None:
            raise NotImplementedError("The road and vehicle must be initialized in the environment implementation")
        # 步数增加：
        self.steps += 1
        # 如果配置了安全保证，通过 safety_supervisor 方法检查并可能修改提供的动作。否则，直接使用原动作。


        if self.config["action_inspector"]:


            self.new_action = self.action_inspector(action)

        else:
            self.new_action = action
            # self.new_action = self._get_available_actions(self.controlled_vehicles,self)

        # action is a tuple, e.g., (2, 3, 0, 1)
        # 调用 _simulate 方法执行动作并推进环境状态。
        # print("self.new_cation===",self.new_action)
        self._simulate(self.new_action)
        # 获取当前观测、计算奖励、判断是否达到终止状态。
        obs = self.observation_type.observe()
        reward = self._reward(action)
        terminal = self._is_terminal()

        # get action masks
        # mynote 是否启用动作mask
        if self.config["action_masking"]:
            # 初始化可用动作列表
            available_actions = [[0] * self.n_a] * len(self.controlled_vehicles)
            # 计算每个车辆的可用动作：
            for i in range(len(self.controlled_vehicles)):
                available_action = self._get_available_actions(self.controlled_vehicles[i], self)
                for a in available_action:
                    # 并在available_actions列表中将这些动作标记为1（可用）。
                    available_actions[i][a] = 1
        else:
            available_actions = [[1] * self.n_a] * len(self.controlled_vehicles)
        # 计算平均速度和收集车辆信息
        for v in self.controlled_vehicles:
            average_speed += v.speed
        #     收集车辆的速度和位置信息，存储在 info 字典中，
        average_speed = average_speed / len(self.controlled_vehicles)

        self.vehicle_speed.append([v.speed for v in self.controlled_vehicles])
        self.vehicle_pos.append(([v.position[0] for v in self.controlled_vehicles]))
        info = {
            "speed": self.vehicle.speed,
            "crashed": self.vehicle.crashed,
            "action": action,
            "new_action": self.new_action,
            "action_mask": np.array(available_actions),
            "average_speed": average_speed,
            "vehicle_speed": np.array(self.vehicle_speed),
            "vehicle_position": np.array(self.vehicle_pos)
        }

        try:
            # 尝试调用 _cost 方法计算当前动作的成本，并将其添加到 info 字典中。如果方法没有实现，则忽略。
            info["cost"] = self._cost(action)
        except NotImplementedError:
            pass

        # print(self.steps)
        # 返回新的观测、奖励、是否结束、以及附加信息的元组。
        return obs, reward, terminal, info

    def _simulate(self, action: Optional[Action] = None) -> None:
        """Perform several steps of simulation with constant action."""
        # 根据配置的模拟频率和策略频率，计算并执行多步模拟。
        for _ in range(int(self.config["simulation_frequency"] // self.config["policy_frequency"])):

            # Forward action to the vehicle
            # 如果提供了动作且不是手动控制模式，每隔一定的时间步将动作应用到车辆上。
            if action is not None \
                    and not self.config["manual_control"] \
                    and self.time % int(self.config["simulation_frequency"] // self.config["policy_frequency"]) == 0:
                self.action_type.act(action)  # defined in action.py
            # 方法更新环境中所有车辆的状态。
            self.road.act()  # Execute an action
            self.road.step(1 / self.config["simulation_frequency"])  # propagate the vehicle state given its actions.
            self.time += 1

            # Automatically render intermediate simulation steps if a viewer has been launched
            # Ignored if the rendering is done offscreen
            # 如果启用了自动渲染，渲染每一步的模拟结果。
            # myrl render
            self._automatic_rendering()
            self.road.clear_vehicle()  # 新增一个函数，清理碰撞的IDM避免堵住出口

            # Stop at terminal states
            # 如果达到了终止状态，提前结束模拟循环
            if self.done or self._is_terminal():
                break
        self.enable_auto_render = False

    # 方法用于渲染仿真环境。它接受一个字符串参数 mode，该参数决定渲染的模式，如人类可视('human')模式或返回像素数组('rgb_array')模式。
    # 方法可以返回一个可选的NumPy数组，当模式为 'rgb_array' 时，返回环境的图像表示。
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """
        Render the environment.

        Create a viewer if none exists, and use it to render an image.
        :param mode: the rendering mode
        """
        # 将传入的渲染模式保存到环境实例变量中。
        self.rendering_mode = mode
        # 如果环境还没有一个视图（viewer）实例，就创建一个新的视图实例。这个视图是渲染环境图像的工具。
        if self.viewer is None:
            self.viewer = EnvViewer(self)
        # 启用自动渲染
        self.enable_auto_render = True

        # If the frame has already been rendered, do nothing
        # 渲染当前帧：
        if self.should_update_rendering:

            self.viewer.display()
        # 如果视图不是在离屏模式下运行，处理与视图相关的交互事件
        if not self.viewer.offscreen:
            self.viewer.handle_events()
        #     如果渲染模式为 'rgb_array'，则从视图获取当前帧的图像数组并返回。
        if mode == 'rgb_array':
            image = self.viewer.get_image()
            return image
        # 渲染完成后，将更新渲染的标志设置为 False，表示当前帧已经渲染完毕。
        self.should_update_rendering = False

    # close 方法用于关闭环境，特别是关闭环境的视图
    def close(self) -> None:
        """
        Close the environment.

        Will close the environment viewer if it exists.
        """
        # 标记环境为结束状态
        self.done = True
        # 如果环境有一个视图实例，调用视图的 close 方法来关闭它，并将视图实例设置为 None，表示当前没有打开的视图。
        if self.viewer is not None:
            self.viewer.close()
        self.viewer = None
    # 它返回当前环境中可用的动作列表。这个方法考虑了车道变化的限制（
    # 比如，车辆位于道路边界时不能变道）和速度变化的限制（在最大或最小速度时不能加速或减速）。
    # nynote 这个函数没用
    def get_available_actions(self) -> List[int]:
        """
        Get the list of currently available actions.

        Lane changes are not available on the boundary of the road, and speed changes are not available at
        maximal or minimal speed.

        :return: the list of available actions
        """
        # 动作类型检查
        if not isinstance(self.action_type, DiscreteMetaAction):
            raise ValueError("Only discrete meta-actions can be unavailable.")
        # 初始化动作列表
        actions = [self.action_type.actions_indexes['IDLE']]
        # 遍历车辆当前车道的相邻车道，根据相对位置和是否可达，决定是否添加向左或向右变道的动作。
        for l_index in self.road.network.side_lanes(self.vehicle.lane_index):
            if l_index[2] < self.vehicle.lane_index[2] \
                    and self.road.network.get_lane(l_index).is_reachable_from(self.vehicle.position) \
                    and self.action_type.lateral:
                actions.append(self.action_type.actions_indexes['LANE_LEFT'])
            if l_index[2] > self.vehicle.lane_index[2] \
                    and self.road.network.get_lane(l_index).is_reachable_from(self.vehicle.position) \
                    and self.action_type.lateral:
                actions.append(self.action_type.actions_indexes['LANE_RIGHT'])
        # 根据车辆当前的速度索引和速度级别的总数，决定是否添加加速或减速的动作。
        # print("sudu index==", self.vehicle.speed_index)
        if self.vehicle.speed_index < self.vehicle.SPEED_COUNT - 1 and self.action_type.longitudinal:
            actions.append(self.action_type.actions_indexes['FASTER'])
        if self.vehicle.speed_index > 0 and self.action_type.longitudinal:
            actions.append(self.action_type.actions_indexes['SLOWER'])
        return actions

    # 在动作执行期间自动渲染中间帧
    def _automatic_rendering(self) -> None:
        """
        Automatically render the intermediate frames while an action is still ongoing.

        This allows to render the whole video and not only single steps corresponding to agent decision-making.

        If a callback has been set, use it to perform the rendering. This is useful for the environment wrappers
        such as video-recording monitor that need to access these intermediate renderings.
        """
        self.should_update_rendering = True
        # 如果有可用的视图（viewer）并且启用了自动渲染，则设置渲染更新标志为真
        if self.viewer is not None and self.enable_auto_render:
            self.should_update_rendering = True

            # 如果设置了自动渲染的回调，则调用该回调函数；否则，直接调用 render 方法进行渲染
            if self.automatic_rendering_callback is not None:
                self.automatic_rendering_callback()
            else:
                # pass
                # myrl render
                self.render(self.rendering_mode)

    # myrl
    # 算到合流终点的距离，别是对于在特定车道汇入口上的车辆，根据它的位置动态计算距离。
    def distance_to_merging_end(self, vehicle):
        # 100
        return 0

    # myrl
    def _compute_headway_distance(self, vehicle):
        # 初始化车头时距为环岛的周长
        # headway_distance = self.road.length
        headway_distance = 60

        # 方法遍历当前车道上的所有车辆，计算并更新车头时距为最小值
        for v in self.road.vehicles:
            if v != vehicle and v.lane_index == vehicle.lane_index:
                # 计算两个车在环岛上的距离
                hd = self._distance_on_roundabout(vehicle, v)

                # 检查车头时距是否小于给定的阈值
                if hd < headway_distance:
                    # 是的话将车头时距设成当前值
                    headway_distance = hd

        return headway_distance

    def _distance_on_roundabout(self, vehicle1, vehicle2):
        # 获取两个车辆在环岛上的位置（角度）
        pos1 = vehicle1.position[1] % self.road.length
        pos2 = vehicle2.position[1] % self.road.length

        # 计算两个车辆在环岛上的距离
        if vehicle1.lane_index == 0:  # 内圈车道
            if pos1 <= pos2:
                distance = pos2 - pos1
            else:
                distance = self.road.length - pos1 + pos2
        else:  # 外圈车道
            if pos1 <= pos2:
                distance = self.road.length - pos2 + pos1
            else:
                distance = pos1 - pos2

        return distance




    # 创建环境的简化副本，去除距离车辆较远的其他车辆。这样做旨在减少决策所需的计算量，同时保留最优动作集。
    def simplify(self) -> 'AbstractEnv':
        """
        Return a simplified copy of the environment where distant vehicles have been removed from the road.
        This is meant to lower the policy computational load while preserving the optimal actions set.

        :return: a simplified environment state
        """
        # 通过 deepcopy 创建环境的完整副本，然后筛选出与 vehicle（自车）
        # 相距一定距离（PERCEPTION_DISTANCE）内的车辆，并仅保留这些车辆。
        state_copy = copy.deepcopy(self)
        state_copy.road.vehicles = [state_copy.vehicle] + state_copy.road.close_vehicles_to(
            state_copy.vehicle, self.PERCEPTION_DISTANCE)
        return state_copy

    # 更改道路上所有车辆的类型
    # 创建环境副本，遍历道路上的所有车辆，除了自车外，将其余车辆的类型更改为由 vehicle_class_path 参数指定的类型。
    def change_vehicles(self, vehicle_class_path: str) -> 'AbstractEnv':
        """
        Change the type of all vehicles on the road

        :param vehicle_class_path: The path of the class of behavior for other vehicles
                             Example: "highway_env.vehicle.behavior.IDMVehicle"
        :return: a new environment with modified behavior model for other vehicles
        """
        vehicle_class = utils.class_from_path(vehicle_class_path)

        env_copy = copy.deepcopy(self)
        vehicles = env_copy.road.vehicles
        for i, v in enumerate(vehicles):
            if v is not env_copy.vehicle:
                vehicles[i] = vehicle_class.create_from(v)
        return env_copy

    # 为所有符合条件的车辆设置首选车道。
    def set_preferred_lane(self, preferred_lane: int = None) -> 'AbstractEnv':
        env_copy = copy.deepcopy(self)
        if preferred_lane:
            for v in env_copy.road.vehicles:
                if isinstance(v, IDMVehicle):
                    v.route = [(lane[0], lane[1], preferred_lane) for lane in v.route]
                    # Vehicle with lane preference are also less cautious
                    v.LANE_CHANGE_MAX_BRAKING_IMPOSED = 1000
        return env_copy

    # 在交叉路口为车辆设置特定的行驶路线
    # 修改环境副本中的 IDMVehicle 类型车辆，为它们在交叉口设置特定的路线
    def set_route_at_intersection(self, _to: str) -> 'AbstractEnv':
        env_copy = copy.deepcopy(self)
        for v in env_copy.road.vehicles:
            if isinstance(v, IDMVehicle):
                v.set_route_at_intersection(_to)
        return env_copy

    # 批量设置车辆的属性。
    # 实现：接收一个包含字段名称和值的元组，遍历环境中的车辆并为它们设置相应的属性。
    def set_vehicle_field(self, args: Tuple[str, object]) -> 'AbstractEnv':
        field, value = args
        env_copy = copy.deepcopy(self)
        for v in env_copy.road.vehicles:
            if v is not self.vehicle:
                setattr(v, field, value)
        return env_copy

    # 目的：调用环境中车辆的方法。
    # 实现：接收一个包含方法名称和参数的元组，遍历环境中的车辆，如果车辆有该方法则调用它。
    def call_vehicle_method(self, args: Tuple[str, Tuple[object]]) -> 'AbstractEnv':
        method, method_args = args
        env_copy = copy.deepcopy(self)
        for i, v in enumerate(env_copy.road.vehicles):
            if hasattr(v, method):
                env_copy.road.vehicles[i] = getattr(v, method)(*method_args)
        return env_copy

    # 随机化 IDMVehicle 类型车辆的行为。
    # 实现：遍历环境副本中的车辆，对 IDMVehicle 类型车辆调用 randomize_behavior 方法，使它们的行为参数随机化。
    def randomize_behaviour(self) -> 'AbstractEnv':
        env_copy = copy.deepcopy(self)
        for v in env_copy.road.vehicles:
            if isinstance(v, IDMVehicle):
                v.randomize_behavior()
        return env_copy

    # 将当前环境转换为有限的马尔可夫决策过程（MDP）。
    # 实现：调用 finite_mdp 函数，传入当前环境和时间量化参数，返回环境对应的有限MDP表示。
    def to_finite_mdp(self):
        return finite_mdp(self, time_quantization=1 / self.config["policy_frequency"])

    # 自定义深拷贝行为，特别是为了避免复制环境中的视图器(viewer)和自动渲染回调(automatic_rendering_callback)。
    def __deepcopy__(self, memo):
        """Perform a deep copy but without copying the environment viewer."""
        cls = self.__class__
        #  创建一个新的实例，但不初始化它
        result = cls.__new__(cls)
        #  字典用于避免在深拷贝过程中重复复制同一对象
        memo[id(self)] = result
        # 遍历原始对象的字典（__dict__），复制所有属性，除了'viewer'和'automatic_rendering_callback'。这两个属性被显式地设置为None
        # ，因为它们通常包含对外部资源的引用，如图形界面组件，这些组件在复制时可能会导致问题或不需要复制。
        for k, v in self.__dict__.items():
            if k not in ['viewer', 'automatic_rendering_callback']:
                setattr(result, k, copy.deepcopy(v, memo))
            else:
                setattr(result, k, None)
        return result


    # myrl
    # 检查一个车辆是否与另一个车辆发生碰撞。
    def check_collision(self, vehicle, other, other_trajectories):
        """
        Check for collision with another vehicle.

        :param other: the other vehicle' trajectories or object
        other_trajectories: [vehicle.position, vehicle.heading, vehicle.speed]
        """
    pass






    def action_dec(obs, info):
        # 初始化一个全1的action数组,大小为obs的长度
        action = np.ones(len(obs))
        # 获取ttc(time to collision)数组
        ttc = np.array(info["ttc"])
        # 如果ttc数组的和为0,将action数组全部设为2
        if sum(ttc) == 0:
            action = action * 2

        else:
            # 否则,找到ttc大于0的最小值对应的智能体索引pri_agent
            pri_agent = np.where(ttc == ttc[ttc > 0].min())[0]
            # 将action[pri_agent]设为2
            action[pri_agent] = 2
            # 遍历每个智能体
            for i in range(len(obs)):
                if i != pri_agent:  # 不是优先的
                    # 如果不是pri_agent,且ttc小于2,且与pri_agent的ttc差小于0.5,将action设为0
                    if ttc[i] < 2 and (ttc[i] - ttc[pri_agent]) < 0.5:
                        action[i] = 0
                #         如果该智能体的out_flag为True,将action设为2
                if info["out_flag"][i]:
                    action[i] = 2
        return tuple(action)


    # myrl 进行具体的碰撞检测。
    def _is_colliding(self, vehicle, other, other_trajectories):
        # Fast spherical pre-check
        # other_trajectories: [vehicle.position, vehicle.heading, vehicle.speed]

        # Euclidean distance
        # 首先，使用简单的球形检测作为预检查，如果两个对象的欧氏距离超过车辆长度，认为它们没有碰撞。
        if np.linalg.norm(other_trajectories[0] - vehicle.position) > vehicle.LENGTH:
            return False

        # Accurate rectangular check
        # 然后，进行更精确的矩形相交检查，考虑车辆的位置、长度、宽度和朝向
        return utils.rotated_rectangles_intersect(
            (vehicle.position, 0.9 * vehicle.LENGTH, 0.9 * vehicle.WIDTH, vehicle.heading),
            (other_trajectories[0], 0.9 * other.LENGTH, 0.9 * other.WIDTH, other_trajectories[1]))

# 定义一个环境的包装器，用于处理多智能体环境中的步骤执行，特别是聚合多个智能体的奖励和结束信号。
class MultiAgentWrapper(Wrapper):
    # 覆盖step方法，首先调用基类的step方法执行动作并获取观测、奖励、结束状态和额外信息。
    def step(self, action):
        obs, reward, done, info = super().step(action)
        reward = np.array(list(info["agents_rewards"]))
        done = np.array(list(info["agents_dones"]))
        return obs, reward, done, info
