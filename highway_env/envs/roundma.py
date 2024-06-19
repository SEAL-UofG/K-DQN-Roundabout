#  TODO 修改reward 函数， 然后速度不能用target speed 速度变化太快不好控制
from typing import Dict, Text

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import CircularLane, LineType, SineLane, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import MDPVehicle
from gym.envs.registration import register
import gym
from gym.spaces import Discrete,Box
from highway_env.road.regulation import RegulatedRoad
# 继承自abstract基类的子类

# myrl
import random
Observation = np.ndarray
from typing import Tuple
from highway_env.vehicle.kinematics import Vehicle

class RoundaboutEnv(AbstractEnv):

    # 动作空间纬度
    n_a = 5
    # 状态空间纬度
    n_s = 25
    @classmethod
    # 参数设置
    def default_config(cls) -> dict:
        config = super().default_config()

        config.update(
            {
                "observation": {
                    "type": "Kinematics",
                    "absolute": True,
                    "features_range": {
                        "x": [-100, 100],
                        "y": [-100, 100],
                        "vx": [-15, 15],
                        "vy": [-15, 15],
                    },
                },
                "action": {"type": "DiscreteMetaAction",
                           "target_speeds": [0, 8],
                           "low_level": "PID"
                           },


                "incoming_vehicle_destination": None,
                "collision_reward": -1,
                "high_speed_reward": 0.2,
                "right_lane_reward": 0,

                "screen_width": 900,
                "screen_height": 900,
                "centering_position": [0.5, 0.6],
                "duration": 20,
                "normalize_reward": True,

                "reward_speed_range": [15, 25],
                "HEADWAY_TIME": 1.2,

                "COLLISION_REWARD": 20,  # default=200
                "HIGH_SPEED_REWARD": 4,  # default=0.5
                "LANE_CHANGE_REWARD": -0.5,#0.05
                "on_road_reward": 1,
                "MERGING_LANE_COST": 4,  # default=4
                "HEADWAY_COST": 4,
                "policy_frequency" : 5,
                "ARRIVE_REWARD": 15,  # 设置一个适当的值
                "HEADWAY_REWARD": 1,
                "reward_reshape": False

            }
        )
        return config








    def _is_terminated(self) -> bool:
        return self.vehicle.crashed



    def _is_truncated(self) -> bool:
        return self.time >= self.config["duration"]

    def _make_road(self) -> None:
        # Circle lanes: (s)outh/(e)ast/(n)orth/(w)est (e)ntry/e(x)it.
        # 定义中心半径和角度
        center = [0, 0]  # [m]
        radius = 20  # [m]
        alpha = 24  # [deg]

        net = RoadNetwork()
        # 创建两圈车道
        radii = [radius, radius + 5]
        # 车道线类型
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        # 连续，虚线，无，连续
        line = [[c, s], [n, c]]


        # 圆环结束4个90

        for lane in [0, 1]:
            # 第一个for循环取0，第二个for循环取1
            # 添加车道线
            # 参数，起始和终点标志

            # 42
            net.add_lane(
                "se",

                "se1",
                CircularLane(

                    center,
                    # 当lane=0时，radi[0]=radius=20
                    radii[lane],
                    # 开始角度和终止角度，66-24
                    np.deg2rad(78),
                    np.deg2rad(66),
                    clockwise=False,
                    line_types=line[lane],
                ),
            )

            net.add_lane(
                "se1",

                "ex1",
                CircularLane(
                    center,
                    # 当lane=0时，radi[0]=radius=20
                    radii[lane],
                    # 开始角度和终止角度，66-24
                    np.deg2rad(66),
                    np.deg2rad(24),
                    clockwise=False,
                    line_types=line[lane],
                ),
            )

            net.add_lane(
                "ex1",

                "ex",
                CircularLane(
                    center,
                    # 当lane=0时，radi[0]=radius=20
                    radii[lane],
                    # 开始角度和终止角度，66-24
                    np.deg2rad(24),
                    np.deg2rad(12),
                    clockwise=False,
                    line_types=line[lane],
                ),
            )

            # 48\

            net.add_lane(
                "ex",
                "ee",
                CircularLane(
                    center,
                    radii[lane],
                    # 24---24，右边水平3点钟方向为0
                    np.deg2rad(12),
                    np.deg2rad(-12),
                    clockwise=False,
                    line_types=line[lane],
                ),
            )
            net.add_lane(
                "ee",
                "ee1",
                CircularLane(
                    center,
                    radii[lane],
                    # 24---24，右边水平3点钟方向为0
                    np.deg2rad(-12),
                    np.deg2rad(-24),
                    clockwise=False,
                    line_types=line[lane],
                ),
            )
            net.add_lane(
                "ee1",
                "nx1",
                CircularLane(
                    center,
                    radii[lane],
                    # 24---24，右边水平3点钟方向为0
                    np.deg2rad(-24),
                    np.deg2rad(-66),
                    clockwise=False,
                    line_types=line[lane],
                ),
            )

            # 42
            net.add_lane(
                "nx1",
                "nx",
                CircularLane(
                    center,
                    radii[lane],
                    # --24-66
                    np.deg2rad(-66),
                    np.deg2rad(-78),
                    clockwise=False,
                    line_types=line[lane],
                ),
            )
            # 逆时阵，-90在12点钟方向，48
            net.add_lane(
                "nx",
                "ne",
                CircularLane(
                    center,
                    radii[lane],
                    np.deg2rad(-78),
                    np.deg2rad(-102),
                    clockwise=False,
                    line_types=line[lane],
                ),
            )

            net.add_lane(
                "ne",
                "ne1",
                CircularLane(
                    center,
                    radii[lane],
                    np.deg2rad(-102),
                    np.deg2rad(-114),
                    clockwise=False,
                    line_types=line[lane],
                ),
            )

            # 42
            net.add_lane(
                "ne1",
                "wx1",
                CircularLane(
                    center,
                    radii[lane],
                    np.deg2rad(-114),
                    np.deg2rad(-156),
                    clockwise=False,
                    line_types=line[lane],
                ),
            )
            # 9点钟方向，48
            net.add_lane(
                "wx1",
                "wx",
                CircularLane(
                    center,
                    radii[lane],
                    np.deg2rad(-156),
                    np.deg2rad(-168),
                    clockwise=False,
                    line_types=line[lane],
                ),
            )
            net.add_lane(
                "wx",
                "we",
                CircularLane(
                    center,
                    radii[lane],
                    np.deg2rad(-168),
                    np.deg2rad(-192),
                    clockwise=False,
                    line_types=line[lane],
                ),
            )

            net.add_lane(
                "we",
                "we1",
                CircularLane(
                    center,
                    radii[lane],
                    np.deg2rad(168),
                    np.deg2rad(156),
                    clockwise=False,
                    line_types=line[lane],
                ),
            )


            # 42
            net.add_lane(
                "we1",
                "sx1",
                CircularLane(
                    center,
                    radii[lane],
                    np.deg2rad(156),
                    np.deg2rad(114),
                    clockwise=False,
                    line_types=line[lane],
                ),
            )

            net.add_lane(
                "sx1",
                "sx",
                CircularLane(
                    center,
                    radii[lane],
                    np.deg2rad(114),
                    np.deg2rad(102),
                    clockwise=False,
                    line_types=line[lane],
                ),
            )
            net.add_lane(
                "sx",
                "se",
                CircularLane(
                    center,
                    radii[lane],
                    np.deg2rad(102),
                    np.deg2rad(78),
                    clockwise=False,
                    line_types=line[lane],
                ),
            )

        #     圆环结束4个90
        #
        # Access lanes: (r)oad/(s)ine
        # 接入长度170m
        access = 85  # [m]
        # 正弦长度85，起点道终点的距离
        # 距离中心的距离
        dev = 85  # [m]
        # 相对中心线的偏移5，分叉的开和程度
        a = 5  # [m]
        # 正弦开始段长度总长度的0.2，弯道长度调节
        delta_st = 0.2 * dev  # [m]
        # 正弦结束长度总长度减去0.2
        delta_en = dev - delta_st
        # 计算了正弦波的角频率
        w = 2 * np.pi / dev
        # 增加车道宽度
        lane_width = 4  # [m]

        # 第一组车道[2, 170]-【2，42.5】
        net.add_lane("ser", "ses", StraightLane([2, access], [2, dev / 2], line_types=(s, c)))
        net.add_lane("ser", "ses", StraightLane([2 + lane_width, access], [2 + lane_width, dev / 2], line_types=(s, c)))

        net.add_lane("ses", "se",
                     SineLane([2 + a, dev / 2], [2 + a, dev / 2 - delta_st], a, w, -np.pi / 2, line_types=(c, c)))
        net.add_lane("ses", "se",
                     SineLane([2 + a + lane_width, dev / 2], [2 + a + lane_width, dev / 2 - delta_st], a, w, -np.pi / 2,
                              line_types=(c, c)))

        net.add_lane("sx", "sxs",
                     SineLane([-2 - a, -dev / 2 + delta_en], [-2 - a, dev / 2], a, w, -np.pi / 2 + w * delta_en,
                              line_types=(c, c)))
        net.add_lane("sx", "sxs",
                     SineLane([-2 - a - lane_width, -dev / 2 + delta_en], [-2 - a - lane_width, dev / 2], a, w,
                              -np.pi / 2 + w * delta_en, line_types=(c, c)))

        net.add_lane("sxs", "sxr", StraightLane([-2, dev / 2], [-2, access], line_types=(n, c)))
        net.add_lane("sxs", "sxr",
                     StraightLane([-2 - lane_width, dev / 2], [-2 - lane_width, access], line_types=(n, c)))

        # 第二组车道
        net.add_lane("eer", "ees", StraightLane([access, -2], [dev / 2, -2], line_types=(s, c)))
        net.add_lane("eer", "ees",
                     StraightLane([access, -2 - lane_width], [dev / 2, -2 - lane_width], line_types=(s, c)))

        net.add_lane("ees", "ee",
                     SineLane([dev / 2, -2 - a], [dev / 2 - delta_st, -2 - a], a, w, -np.pi / 2, line_types=(c, c)))
        net.add_lane("ees", "ee",
                     SineLane([dev / 2, -2 - a - lane_width], [dev / 2 - delta_st, -2 - a - lane_width], a, w,
                              -np.pi / 2, line_types=(c, c)))

        net.add_lane("ex", "exs",
                     SineLane([-dev / 2 + delta_en, 2 + a], [dev / 2, 2 + a], a, w, -np.pi / 2 + w * delta_en,
                              line_types=(c, c)))
        net.add_lane("ex", "exs",
                     SineLane([-dev / 2 + delta_en, 2 + a + lane_width], [dev / 2, 2 + a + lane_width], a, w,
                              -np.pi / 2 + w * delta_en, line_types=(c, c)))

        net.add_lane("exs", "exr", StraightLane([dev / 2, 2], [access, 2], line_types=(n, c)))
        net.add_lane("exs", "exr", StraightLane([dev / 2, 2 + lane_width], [access, 2 + lane_width], line_types=(n, c)))

        # 第三组车道
        net.add_lane("ner", "nes", StraightLane([-2, -access], [-2, -dev / 2], line_types=(s, c)))
        net.add_lane("ner", "nes",
                     StraightLane([-2 - lane_width, -access], [-2 - lane_width, -dev / 2], line_types=(s, c)))

        net.add_lane("nes", "ne",
                     SineLane([-2 - a, -dev / 2], [-2 - a, -dev / 2 + delta_st], a, w, -np.pi / 2, line_types=(c, c)))
        net.add_lane("nes", "ne",
                     SineLane([-2 - a - lane_width, -dev / 2], [-2 - a - lane_width, -dev / 2 + delta_st], a, w,
                              -np.pi / 2, line_types=(c, c)))

        net.add_lane("nx", "nxs",
                     SineLane([2 + a, dev / 2 - delta_en], [2 + a, -dev / 2], a, w, -np.pi / 2 + w * delta_en,
                              line_types=(c, c)))
        net.add_lane("nx", "nxs",
                     SineLane([2 + a + lane_width, dev / 2 - delta_en], [2 + a + lane_width, -dev / 2], a, w,
                              -np.pi / 2 + w * delta_en, line_types=(c, c)))

        net.add_lane("nxs", "nxr", StraightLane([2, -dev / 2], [2, -access], line_types=(n, c)))
        net.add_lane("nxs", "nxr",
                     StraightLane([2 + lane_width, -dev / 2], [2 + lane_width, -access], line_types=(n, c)))

        # 第四组车道
        net.add_lane("wer", "wes", StraightLane([-access, 2], [-dev / 2, 2], line_types=(s, c)))
        net.add_lane("wer", "wes",
                     StraightLane([-access, 2 + lane_width], [-dev / 2, 2 + lane_width], line_types=(s, c)))

        net.add_lane("wes", "we",
                     SineLane([-dev / 2, 2 + a], [-dev / 2 + delta_st, 2 + a], a, w, -np.pi / 2, line_types=(c, c)))
        net.add_lane("wes", "we",
                     SineLane([-dev / 2, 2 + a + lane_width], [-dev / 2 + delta_st, 2 + a + lane_width], a, w,
                              -np.pi / 2, line_types=(c, c)))

        net.add_lane("wx", "wxs",
                     SineLane([dev / 2 - delta_en, -2 - a], [-dev / 2, -2 - a], a, w, -np.pi / 2 + w * delta_en,
                              line_types=(c, c)))
        net.add_lane("wx", "wxs",
                     SineLane([dev / 2 - delta_en, -2 - a - lane_width], [-dev / 2, -2 - a - lane_width], a, w,
                              -np.pi / 2 + w * delta_en, line_types=(c, c)))

        net.add_lane("wxs", "wxr", StraightLane([-dev / 2, -2], [-access, -2], line_types=(n, c)))
        net.add_lane("wxs", "wxr",
                     StraightLane([-dev / 2, -2 - lane_width], [-access, -2 - lane_width], line_types=(n, c)))

        road = RegulatedRoad(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )
        self.road = road

    # mynote 这里的生成的车和前面的road必须一致，车道线必须是一个闭环首尾相连
    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.
        :return: the ego-vehicle
        """
        # MYRL
        self.controlled_vehicles = []

        # 定义车辆位置和速度的随机偏差值
        position_deviation = 2
        speed_deviation = 2
        self.road.controlled_vehicles = []

        # mynote Ego-vehicle1
        # 获取道路网络上的特定车道的引用这里是六点车道
        ego_lane1 = self.road.network.get_lane(("ser", "ses", 1))
        # 创建一个新的自主车辆实例
        ego_vehicle1 = self.action_type.vehicle_class(
            self.road,
            # 位置
            ego_lane1.position(30, 0),
            # 速度
            speed=23,
            heading=ego_lane1.heading_at(140),
        )
        try:
            # print( ego_vehicle1)
            ego_vehicle1.plan_route_to(ego_vehicle1,"wxr", 0)
            ego_vehicle1.mpc_dt = self.config["simulation_frequency"]
        except AttributeError:
            pass
        ego_vehicle1.vehicle_id = 0  # 保存车辆编号
        self.road.vehicles.append(ego_vehicle1)
        self.road.controlled_vehicles.append(ego_vehicle1)
        self.controlled_vehicles.append(ego_vehicle1)

        # mynote Ego-vehicle2
        # ego_lane2 = self.road.network.get_lane(("wer1", "wes1", 0))
        # # 创建一个新的自主车辆实例
        # ego_vehicle2 = self.action_type.vehicle_class(
        #     self.road,
        #     # 位置
        #     ego_lane2.position(125, 0),
        #     # 速度
        #     speed=15,
        #     heading=ego_lane2.heading_at(140),
        # )
        # try:
        #     # # 为自主车辆规划目的地 wxs9点的上线
        #     # vehicle_count_inner = self.road.get_vehicle_count(lane_index=0)
        #     # vehicle_count_outer = self.road.get_vehicle_count(lane_index=1)
        #     # # Choose lane based on traffic distribution and proximity of vehicles in different lanes
        #     # if vehicle_count_inner >= vehicle_count_outer:
        #     #     chosen_lane2 = 1  # Prefer inner lane if outer is more crowded or nearest vehicle is on outer lane
        #     # else:
        #     #     chosen_lane2 = 0  # Prefer outer lane if inner is more crowded
        #     # # 为自主车辆规划目的地 wxs9点的上线
        #     ego_vehicle2.plan_route_to("exr", 0)
        #     ego_vehicle2.destination_name = {"name": "wxr", "lane": 0}  # 使用字典存储信息
        #     # print("ego_vehicle.lane_index===",ego_vehicle.lane_index)
        # except AttributeError:
        #     pass
        # ego_vehicle2.vehicle_id = 1  # 保存车辆编号
        # self.road.vehicles.append(ego_vehicle2)
        # self.road.controlled_vehicles.append(ego_vehicle2)
        # self.controlled_vehicles.append(ego_vehicle2)

        # ... (rest of the function remains the same)

        # mynote Ego-vehicle 3
        # ego_lane = self.road.network.get_lane(("wer", "wes", 0))
        # # 创建一个新的自主车辆实例
        # ego_vehicle = self.action_type.vehicle_class(
        #     self.road,
        #     # 位置
        #     ego_lane.position(100, 0),
        #     # 速度
        #     speed=20,
        #     heading=ego_lane.heading_at(140),
        # )
        # try:
        #     # 为自主车辆规划目的地 wxs9点的上线
        #     ego_vehicle.plan_route_to("exs", 0)
        # except AttributeError:
        #     pass
        # self.road.vehicles.append(ego_vehicle)
        # self.vehicle = ego_vehicle

        # Incoming vehicle
        destinations = ["wxr","sxr"]
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        # print("11111111111111====",self.config["other_vehicles_type"])

        vehicle = other_vehicles_type.make_on_lane(
            self.road,
            ("wx", "we", 1),
            longitudinal=-10 ,
            speed=20 ,
        )
        # print("vehicle===",vehicle.route)

        # if self.config["incoming_vehicle_destination"] is not None:
        #     destination = destinations[self.config["incoming_vehicle_destination"]]
        # else:
        destination = np.random.choice(destinations)
        # print(f"Vehicle type: {type(vehicle)}")
        # print(f"Vehicle: {vehicle}")
        vehicle.plan_route_to1(destination,0)
        vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)

        # Other vehicles
        # 修改这里增加车辆
        for i in list(range(0, 2)):

            vehicle = other_vehicles_type.make_on_lane(
                self.road,
                ("we", "we1", i),
                longitudinal=30 * i + np.random.normal() * position_deviation,
                speed=20 + np.random.normal() * speed_deviation,
            )
            vehicle.plan_route_to1(np.random.choice(destinations),i)
            # vehicle.randomize_behavior()
            self.road.vehicles.append(vehicle)

        for i in list(range(0, 2)):
            vehicle = other_vehicles_type.make_on_lane(
                self.road,
                ("ee1", "nx1", i),
                longitudinal=20 * i + np.random.normal() * position_deviation,
                speed=20 + np.random.normal() * speed_deviation,
            )
            vehicle.plan_route_to1(np.random.choice(destinations), i)
            # vehicle.randomize_behavior()
            self.road.vehicles.append(vehicle)

        for i in list(range(0, 2)):
            vehicle = other_vehicles_type.make_on_lane(
                self.road,
                ("se1", "ex1", i),
                longitudinal=20 * i + np.random.normal() * position_deviation,
                speed=20 + np.random.normal() * speed_deviation,
            )
            vehicle.plan_route_to1(np.random.choice(destinations), i)
            # vehicle.randomize_behavior()
            self.road.vehicles.append(vehicle)

        # Entering vehicle
        # 随机创建一个车在3点钟上面车道往里进
        self.lane_index =0
        vehicle = other_vehicles_type.make_on_lane(
            self.road,
            ("eer", "ees", self.lane_index),
            longitudinal=120 + np.random.normal() * position_deviation,
            speed=20 + np.random.normal() * speed_deviation,
        )
        vehicle.plan_route_to1(np.random.choice(destinations),self.lane_index)
        # vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)
        # 随机创建一个车在12点钟上面车道往里进
        self.lane_index = 0
        vehicle = other_vehicles_type.make_on_lane(
            self.road,
            ("ner", "nes",self.lane_index),
            longitudinal=120,
            speed=20,
        )
        vehicle.plan_route_to1(np.random.choice(destinations),self.lane_index)
        # vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)
        # 随机创建一个车在9点钟上面车道往里进
        # self.lane_index = 0
        # vehicle = other_vehicles_type.make_on_lane(
        #     self.road,
        #     ("wer", "wes", 1),
        #     longitudinal=125 + np.random.normal() * position_deviation,
        #     speed=20 + np.random.normal() * speed_deviation,
        # )
        # vehicle.0("exr",1)
        # # vehicle.randomize_behavior()
        # self.road.vehicles.append(vehicle)

        # 随机创建一个车在9点钟上面车道往里进
        # self.lane_index = 0
        # vehicle = other_vehicles_type.make_on_lane(
        #     self.road,
        #     ("wer", "wes", 0),
        #     longitudinal=100 + np.random.normal() * 5,
        #     speed=20 + np.random.normal() * speed_deviation,
        # )
        # vehicle.plan_route_to("exr", 0)
        # # vehicle.randomize_behavior()
        # self.road.vehicles.append(vehicle)

        # ###################################################################
        #
        # 检查每辆控制车辆是否到达距离终点一半的位置
        # for vehicle in self.controlled_vehicles:
        #     print("vehicle.lane_index[1]===",vehicle.lane_index[1])
        #     print("vehicle.destination_name===",vehicle.destination_name["name"])
        #     if vehicle.lane_index[1] == vehicle.destination_name["name"]:
        #
        #         vehicle.is_arrive = True
        #     else:
        #         vehicle.is_arrive = False

    def _reward(self, action: list) -> float:

        for vehicle in self.controlled_vehicles:
            # print(vehicle.lane_index[1])
            if vehicle.lane_index[1] == "exs" or vehicle.lane_index[1] == "nxs"\
                or vehicle.lane_index[1] == "wxs"  or vehicle.lane_index[1] == "sxs":


                vehicle.is_arrive = True
                # print(vehicle.is_arrive)
                # print(vehicle.is_arrive)
            else:
                vehicle.is_arrive = False


        # Cooperative multi-agent reward
        # myrl 这里的是元组
        return sum(self._agent_reward(action, vehicle) for vehicle in self.controlled_vehicles) \
            / len(self.controlled_vehicles)


    def _reset(self, num_CAV=0) -> None:
        self._make_road()

        #     调用_make_vehicles方法来生成这些车辆
        self._make_vehicles()

        # 环境状态中的一些其他属性被设置或重置
        self.action_is_safe = True
        # T是根据配置中的duration和policy_frequency计算得出的仿真步数
        # ，用于控制仿真的持续时间。
        self.T = int(self.config["duration"] * self.config["policy_frequency"])
        # myrl render
        self.render()
    #  创建道路环境
    def _is_terminal(self) -> bool:
        # self.steps 来自与父类
        # print(all(self.has_arrived(vehicle) for vehicle in self.controlled_vehicles))
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return any(vehicle.crashed for vehicle in self.controlled_vehicles) \
               or self.steps >= self.config["duration"] * self.config["policy_frequency"] \
                or all(self.has_arrived(vehicle) for vehicle in self.controlled_vehicles)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        car_count = len([vehicle for vehicle in self.road.vehicles if isinstance(vehicle, MDPVehicle)])
        # print(car_count)
        # print("vehicle position==", self.vehicle.position)
        # 首先创建了一个空列表agent_info用于后续收集受控车辆的信息
        agent_info = []
        # 调用父类的step方法来执行给定的动作，并获取基本的仿真更新，包括新的观测、奖励、是否完成的标志和附加信息
        obs, reward, done, info = super().step(action)
        #计算每辆受控车辆是否达到终止状态，并将这些信息添加到info字典中。这对于了解仿真中每个代理的状态非常有用。
        info["agents_dones"] = tuple(self._agent_is_terminal(vehicle) for vehicle in self.controlled_vehicles)
        # 这里遍历所有受控车辆，收集它们的位置和速度信息，然后将这些信息添加到info字典中
        for v in self.controlled_vehicles:
            agent_info.append([v.position[0], v.position[1], v.speed])
        info["agents_info"] = agent_info
        # 对每辆受控车辆，根据执行的动作和车辆的状态计算局部奖励，并更新到车辆的状态中。
        for vehicle in self.controlled_vehicles:
            # myrl 这里是元组
            vehicle.local_reward = self._agent_reward(action, vehicle)
        # local reward
        # 将所有受控车辆的局部奖励收集到一个元组中，并添加到info字典中
        info["agents_rewards"] = tuple(vehicle.local_reward for vehicle in self.controlled_vehicles)
        # regional reward
        # 调用_regional_reward方法计算区域奖励，并将这些奖励添加到info字典中。
        self._regional_reward()
        info["regional_rewards"] = tuple(vehicle.regional_reward for vehicle in self.controlled_vehicles)
        # 将观测值转换为NumPy数组，并重新调整形状，以适应后续处理。
        obs = np.asarray(obs).reshape((len(obs), -1))
        # 返回更新后的观测、奖励、完成标志和附加信息。
        return obs, reward, done, info








    # myrl 拟奖励设置
    def virtual_reward(self, action: int) -> float:
        # 查看虚拟道路上的虚拟奖励
        # myrl 这里是字典
        return sum(self._agent_reward(tuple(action), vehicle) for vehicle in self.virtual_road.controlled_vehicles
                   ) / len(self.virtual_road.controlled_vehicles)

    # # myrl 的虚拟info
    # 定义虚拟道路上的受控车辆的一些关键信息，比如是否碰撞、各个代理的奖励、
    # 是否完成了任务（比如是否到达目的地或发生碰撞）、TTC（时间到碰撞）等
    def virtual_info(self, action: int) -> dict:
        info = {'crashed': tuple(vehicle.crashed for vehicle in self.virtual_road.controlled_vehicles),
                "agents_rewards": tuple(
                    self._agent_reward(action, vehicle) for vehicle in self.virtual_road.controlled_vehicles),
                "agents_dones": tuple(
                    self._agent_is_terminal(vehicle) for vehicle in self.virtual_road.controlled_vehicles),
                'all_arrive': tuple(self.has_arrived(vehicle) for vehicle in self.virtual_road.controlled_vehicles),
                # "in_flag": tuple(vehicle.in_flag for vehicle in self.virtual_road.controlled_vehicles),
                # "out_flag": tuple(vehicle.in_flag for vehicle in self.virtual_road.controlled_vehicles),
                "ttc": tuple(vehicle.TTC for vehicle in self.virtual_road.controlled_vehicles)}
        return info

    def has_arrived(self, vehicle: Vehicle, exit_distance: float = 30) -> bool:
        if vehicle.is_arrive:
            return True
        # 车辆在退出车道并且距离车道末端的距离小于exit_distance，
        # 则认为车辆已经到达目的地，设置vehicle.is_arrive为真并返回真。
        # 否则，设置为假并返回假
        else:

            return False

    def _agent_is_terminal(self, vehicle: Vehicle) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return vehicle.crashed \
            or self.steps >= self.config["duration"] * self.config["policy_frequency"]\
            or all(self.has_arrived(vehicle) for vehicle in self.controlled_vehicles)

    def _agent_reward(self, action: int, vehicle: Vehicle) -> float:
        """
        The vehicle is rewarded for driving with high speed on lanes and avoiding collisions
        :param action: the action performed
        :return: the reward of the state-action transition
        """

        # 计算车辆速度与设定速度范围的线性映射值
        scaled_speed = utils.lmap(vehicle.speed, self.config["reward_speed_range"], [0, 1])

        # 计算相对速度差
        relative_speed = vehicle.speed - np.mean([v.speed for v in self.road.vehicles if v != vehicle])

        # 根据相对速度差调整速度奖励
        high_speed_reward = np.clip(scaled_speed, 0, 1) * np.exp(-0.5 * (relative_speed / vehicle.MAX_SPEED) ** 2)

        # 计算车头时距成本
        v_fr, v_rr, v_fl, v_rl = self.road.surrounding_vehicles(vehicle)
        distances = []

        if v_fr is not None:
            distances.append(np.linalg.norm(v_fr.position - vehicle.position))
        if v_fl is not None:
            distances.append(np.linalg.norm(v_fl.position - vehicle.position))
        if v_rr is not None:
            distances.append(np.linalg.norm(v_rr.position - vehicle.position))
        if v_rl is not None:
            distances.append(np.linalg.norm(v_rl.position - vehicle.position))

        if distances:
            distance = min(distances)
        else:
            distance = 15  # 如果所有值都为 None,将距离设置为无穷大
        # print("距离",distance)

        # 判断车辆与环岛上最近车辆的距离是否大于10米
        if distance > 10:
            headway_reward = 1.0
        else:
            headway_reward = 0.0

        # 计算变道成本
        lane_change_cost = 0
        if isinstance(action, tuple):
            if any(np.any(a == 0) or np.any(a == 2) for a in action):
                lane_change_cost = -0.5
        else:
            if np.any(action == 0) or np.any(action == 2):
                lane_change_cost = -0.5

        # 计算各个奖励组成部分的值
        rewards = {
            "collision_reward": self.config["COLLISION_REWARD"] * (-1 * vehicle.crashed),  # [-1,0]*200
            "high_speed_reward": self.config["HIGH_SPEED_REWARD"] * high_speed_reward,  # 最大似乎也是1
            "lane_change_reward": self.config["LANE_CHANGE_REWARD"] * lane_change_cost,  # [-0.5,0]
            "on_road_reward": int(vehicle.on_road),  # [0,1]
            "merging_lane_cost": self.config["MERGING_LANE_COST"] * 0,  # 0
            "headway_reward": self.config["HEADWAY_REWARD"] * headway_reward,  # [1,0]*1
            "arrive_reward": self.config["ARRIVE_REWARD"] * vehicle.is_arrive  # [1,0]*10
        }
        low = self.config["LANE_CHANGE_REWARD"] * (-0.5)
        high = self.config["ARRIVE_REWARD"] + self.config["HEADWAY_REWARD"] + 1 + self.config["HIGH_SPEED_REWARD"] * 1

        reward = sum(rewards.values())
        reward = utils.lmap(reward, [low, high], [0, 1]) + rewards["collision_reward"] if self.config[
            "reward_reshape"] else reward
        return reward



    def _regional_reward(self):
        # 遍历所有受控车辆。
        for vehicle in self.controlled_vehicles:
            # 为当前车辆初始化一个空列表，用于存储其周围的车辆。
            neighbor_vehicle = []


            v_fr, v_rr,v_fl,v_rl = self.road.surrounding_vehicles(vehicle)


            # 将检测到的周围车辆（如果它们是MDPVehicle类型）添加到neighbor_vehicle列表中。
            for v in [v_fl, v_fr, vehicle, v_rl, v_rr]:
                if type(v) is MDPVehicle and v is not None:
                    neighbor_vehicle.append(v)
            # 计算区域性奖励，即所有周围车辆的局部奖励之和，然后将这个总和平均分配给每辆车
            # 周围没车,这个值为0
            # 区域内的奖励总值
            regional_reward = sum(v.local_reward for v in neighbor_vehicle)
            # 每个车的平均奖励
            # 保留neighbor_vehicle里面不为none的元素，计算元素值
            # print("reginal_reward==",regional_reward / sum(1 for _ in filter(None.__ne__, neighbor_vehicle)))
            vehicle.regional_reward = regional_reward / sum(1 for _ in filter(None.__ne__, neighbor_vehicle))

class RoundaboutMARL(RoundaboutEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "action": {
                "type": "MultiAgentAction",
                "action_config": {
                    "type": "DiscreteMetaAction",
                    "lateral": True,
                    "longitudinal": True,
                    "low_level": "PID"
                }},
            "observation": {
                "type": "MultiAgentObservation",
                "observation_config": {
                    "type": "Kinematics"
                }},
            "controlled_vehicles": 2
        })
        return config

# myrl 为了注册
register(
    id='round-v2',
    entry_point='highway_env.envs.roundma:RoundaboutEnv',
)
register(
    id='round-multi-v2',
    entry_point='highway_env.envs.roundma:RoundaboutMARL',
)
#