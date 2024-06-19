import copy
from typing import List, Optional, Tuple, Union

import numpy as np

from highway_env import utils
from highway_env.road.road import LaneIndex, Road, Route
from highway_env.utils import Vector
from highway_env.vehicle.kinematics import Vehicle

# mychange
# 创建一个虚拟的输出流来捕获所有输出
import casadi as ca
import io
import sys
class DummyFile(io.StringIO):
    def write(self, x): pass

    def flush(self): pass


class ControlledVehicle(Vehicle):
    """
    A vehicle piloted by two low-level controller, allowing high-level actions such as cruise control and lane changes.

    - The longitudinal controller is a speed controller;
    - The lateral controller is a heading controller cascaded with a lateral position controller.
    """

    target_speed: float
    """ Desired velocity."""

    """Characteristic time"""
    TAU_ACC = 0.6  # [s]
    TAU_HEADING = 0.2  # [s]
    TAU_LATERAL = 0.6  # [s]

    TAU_PURSUIT = 0.5 * TAU_HEADING  # [s]
    KP_A = 1 / TAU_ACC
    KP_HEADING = 1 / TAU_HEADING
    KP_LATERAL = 1 / TAU_LATERAL  # [1/s]
    MAX_STEERING_ANGLE = np.pi / 3  # [rad]
    DELTA_SPEED = 5  # [m/s]
    # MYCHANGE PID的最大加速度
    MAX_A = 5
    N = 5  # mpc step
    mpc_dt = 1 / 5  # mpc dt
    min_distance = 7  # 所允许的最小距离

    def __init__(
        self,
        road: Road,
        position: Vector,
        heading: float = 0,
        speed: float = 0,
        target_lane_index: LaneIndex = None,
        target_speed: float = None,
        route: Route = None,
    ):
        super().__init__(road, position, heading, speed)
        self.target_lane_index = target_lane_index or self.lane_index
        self.target_speed = target_speed or self.speed
        self.route = route

        #     myrl
        self.is_route_initialized = False
        self.initial_distance = None  # 初始距离
        self.is_distance_initialized = False
        # mychange
        self.low_controller = None

        self.re_plan = False

    @classmethod
    def create_from(cls, vehicle: "ControlledVehicle") -> "ControlledVehicle":
        """
        Create a new vehicle from an existing one.

        The vehicle dynamics and target dynamics are copied, other properties are default.

        :param vehicle: a vehicle
        :return: a new vehicle at the same dynamical state
        """
        v = cls(
            vehicle.road,
            vehicle.position,
            heading=vehicle.heading,
            speed=vehicle.speed,
            target_lane_index=vehicle.target_lane_index,
            target_speed=vehicle.target_speed,
            route=vehicle.route,
        )
        return v
    #  TODO 这里short的车应该用 surround vehicles 而不是self.road vehicle
    def plan_route_to(self, control_vehicle, destination: str = "o1", lane_index: int = 0) -> "ControlledVehicle":
        """
        Plan a route to a destination in the road network

        :param destination: a node in the road network
        """
        try:
            path = self.road.network.shortest_path1(self.lane_index[1], destination)
        except KeyError:
            path = []

        if path:
            self.route = [self.lane_index]
            for i in range(len(path) - 1):
                _from = path[i]
                _to = path[i + 1]

                v_fr, v_rr, v_fl, v_rl = self.road.surrounding_vehicles(control_vehicle)
                list_sur_v = [v_fr, v_rr, v_fl, v_rl]

                # Calculate traffic density for inner and outer lanes
                inner_density = self.calculate_traffic_density(control_vehicle, _to, 0, self.road.vehicles)
                # print("inner_density===",inner_density)
                outer_density = self.calculate_traffic_density(control_vehicle, _to, 1, self.road.vehicles)
                # print("outer_density===", outer_density)

                # Calculate lane change cost for inner and outer lanes
                inner_cost = self.calculate_lane_change_cost(control_vehicle, _to, 0, self.road.vehicles)
                # print("inner_cost===", inner_cost)
                outer_cost = self.calculate_lane_change_cost(control_vehicle, _to, 1,self.road.vehicles)
                # print("outer_cost===", outer_cost)

                # Choose the lane with lower traffic density and lane change cost

                if inner_density + inner_cost < outer_density + outer_cost:

                    self.route.append((_from, _to, 0))
                    self.re_plan = True
                elif inner_density + inner_cost == outer_density + outer_cost: #and control_vehicle.position[1] > 5:
                    for vehicle in self.road.vehicles:
                        if vehicle is not None and vehicle is not control_vehicle:
                            if vehicle.lane_index[0] == _from and vehicle.lane_index[2] == 1- lane_index and vehicle.lane_index == 0:
                                self.route.append((_from, _to, 0))
                            elif vehicle.lane_index[0] == _from and vehicle.lane_index[2] == 1- lane_index and vehicle.lane_index == 1:
                                self.route.append((_from, _to, 1))


                else:
                    self.route.append((_from, _to, 1))
                    self.re_plan = True

            # print("self.route===", self.route)
        else:
            self.route = [self.lane_index]

        return self



    def calculate_traffic_density(self, control_vehicle, node: str, lane_index: int, vehicles: List['Vehicle']) -> float:
        """
        Calculate the traffic density at a given node and lane index.

        :param node: the node to calculate traffic density
        :param lane_index: the lane index (0 for inner lane, 1 for outer lane)
        :param vehicles: list of vehicles on the road
        :return: traffic density at the given node and lane index
        """
        density = 0.0
        for vehicle in vehicles:
            if vehicle is not None and vehicle is not control_vehicle:
                # 前面正车道
                if vehicle.lane_index[1] == node and vehicle.lane_index[2] == lane_index:
                    density += 1#1
                ##当前车的旁车车道, 密度越大这个道越不用,旁边道有车,这个道应该用
                elif vehicle.lane_index[0] == node and vehicle.lane_index[2] == 1- lane_index:
                    density -= 1  # 1
        return density



    def calculate_lane_change_cost(self, control_vehicle: 'Vehicle', node: str, lane_index: int,
                                   vehicles: List['Vehicle']) -> float:
        """
        Calculate the cost of changing lanes.

        :param control_vehicle: the vehicle for which the cost is calculated
        :param node: current node
        :param lane_index: current lane index
        :param vehicles: list of vehicles on the road
        :return: cost of changing lanes
        """
        cost = 0.0
        for vehicle in vehicles:
            # 前车车道
            if vehicle is not None and vehicle is not control_vehicle:
                if vehicle.lane_index[1] == node and vehicle.lane_index[2] == lane_index:
                    distance = np.linalg.norm(vehicle.position - control_vehicle.position)
                    if distance < 20:
                        cost += 15.0 / distance
                #当前车的旁车车道
                elif vehicle.lane_index[0] == node and vehicle.lane_index[2] == 1-lane_index:
                    distance = np.linalg.norm(vehicle.position - control_vehicle.position)
                    if distance < 10:
                        cost -= 15.0 / distance
        return cost



    def plan_route_to1(self, destination: str= "o1",lane_index:int=0) -> "ControlledVehicle":
        """
        Plan a route to a destination in the road network

        :param destination: a node in the road network
        """
        try:
            path = self.road.network.shortest_path1(self.lane_index[1], destination)
            # print(path)
        except KeyError:
            path = []
        if path:
            # self.route = [self.lane_index] + [
            #     (path[i], path[i + 1], None) for i in range(len(path) - 1)
            # ]
            # mychange none-》0
            # print("lane_index====",lane_index)
            self.route = [self.lane_index] + [(path[i], path[i + 1], lane_index) for i in range(len(path) - 1)]
            # print("IDM", self.route)
            # if lane_index == 1:
            #     # print("self,rout===",self.route)
        else:
            self.route = [self.lane_index]
        return self

    # def replan_route(self):
    #     if self.re_plan:
    #         if self.lane_index[1] != "wxr":
    #             self.plan_route_to(self, "wxr", 0)








    def act(self, action: Union[dict, str] = None) -> None:
        """
        Perform a high-level action to change the desired lane or speed.

        - If a high-level action is provided, update the target speed and lane;
        - then, perform longitudinal and lateral control.

        :param action: a high-level action
        """
        # if self.road and self.road.vehicles:
        #     self.plan_route_to("wxr", 0)

        # if self.lane_index[1] != "wxr":
        #     if self.lane_index[1] != "wx1":
        #         self.plan_route_to(self, "wxr", 0)
        if self.lane_index[1] != "wxr" and self.position[1]>0:

            self.plan_route_to(self, "wxr", 0)

        self.follow_road()
        if action == "FASTER":
            self.target_speed += self.DELTA_SPEED
        elif action == "SLOWER":
            self.target_speed -= self.DELTA_SPEED
        elif action == "LANE_RIGHT":
            _from, _to, _id = self.target_lane_index
            target_lane_index = (
                _from,
                _to,
                np.clip(_id + 1, 0, len(self.road.network.graph[_from][_to]) - 1),
            )
            if self.road.network.get_lane(target_lane_index).is_reachable_from(
                self.position
            ):
                self.target_lane_index = target_lane_index
        elif action == "LANE_LEFT":
            _from, _to, _id = self.target_lane_index
            target_lane_index = (
                _from,
                _to,
                np.clip(_id - 1, 0, len(self.road.network.graph[_from][_to]) - 1),
            )
            if self.road.network.get_lane(target_lane_index).is_reachable_from(
                self.position
            ):
                self.target_lane_index = target_lane_index

        action = {
            "steering": self.steering_control(self.target_lane_index),
            # "acceleration": self.speed_control(self.target_speed),
            # mychange
            "acceleration": self.MPC_controller_for_speed(
                self.target_speed) if self.low_controller == "MPC" else self.speed_control(self.target_speed)
        }
        # print("ACT-self==",self.low_controller)
        action["steering"] = np.clip(
            action["steering"], -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE
        )
        super().act(action)

    def follow_road(self) -> None:
        """At the end of a lane, automatically switch to a next one."""
        if self.road.network.get_lane(self.target_lane_index).after_end(self.position):
            self.target_lane_index = self.road.network.next_lane(
                self.target_lane_index,
                route=self.route,
                position=self.position,
                np_random=self.road.np_random,
            )

    def steering_control(self, target_lane_index: LaneIndex) -> float:
        """
        Steer the vehicle to follow the center of an given lane.

        1. Lateral position is controlled by a proportional controller yielding a lateral speed command
        2. Lateral speed command is converted to a heading reference
        3. Heading is controlled by a proportional controller yielding a heading rate command
        4. Heading rate command is converted to a steering angle

        :param target_lane_index: index of the lane to follow
        :return: a steering wheel angle command [rad]
        """
        target_lane = self.road.network.get_lane(target_lane_index)
        lane_coords = target_lane.local_coordinates(self.position)
        lane_next_coords = lane_coords[0] + self.speed * self.TAU_PURSUIT
        lane_future_heading = target_lane.heading_at(lane_next_coords)

        # Lateral position control
        lateral_speed_command = -self.KP_LATERAL * lane_coords[1]
        # Lateral speed to heading
        heading_command = np.arcsin(
            np.clip(lateral_speed_command / utils.not_zero(self.speed), -1, 1)
        )
        heading_ref = lane_future_heading + np.clip(
            heading_command, -np.pi / 4, np.pi / 4
        )
        # Heading control
        heading_rate_command = self.KP_HEADING * utils.wrap_to_pi(
            heading_ref - self.heading
        )
        # Heading rate to steering angle
        slip_angle = np.arcsin(
            np.clip(
                self.LENGTH / 2 / utils.not_zero(self.speed) * heading_rate_command,
                -1,
                1,
            )
        )
        steering_angle = np.arctan(2 * np.tan(slip_angle))
        steering_angle = np.clip(
            steering_angle, -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE
        )
        return float(steering_angle)

    def speed_control(self, target_speed: float) -> float:
        """
        Control the speed of the vehicle.

        Using a simple proportional controller.

        :param target_speed: the desired speed
        :return: an acceleration command [m/s2]
        """
        return self.KP_A * (target_speed - self.speed)

    # mychange
    def MPC_controller_for_speed(self, target_speed: float) -> float:
        # return self.KP_A * (target_speed - self.speed)
        # print("MPC")
        success = False
        for i in range(self.MAX_A + 1):
            ego_vehicle = copy.deepcopy(self)  # 不影响真实的情况，仅做推演
            observation_vehicle = [copy.deepcopy(vehicles) for vehicles in self.road.vehicles if
                                   0 < np.linalg.norm(ego_vehicle.position - vehicles.position, 2) < 200]

            for vehicle in observation_vehicle:
                vehicle.DELTA = 4  # 这个是一个随机的策略，但你不知道，所以你需要假设他是固定的某个值，控制器不应该有这个信息

            # 创建 Opti 对象
            opti = ca.Opti()

            u = opti.variable(self.N)
            objective = 0
            for t in range(self.N):
                # 每一个step，都先act再step
                for vehicle in observation_vehicle:
                    vehicle.act()  # 根据对它们策略的推测来演化得到加速度和舵机转向
                for vehicle in observation_vehicle:
                    vehicle.step(1 / self.mpc_dt)
                # 舵机用原来的，缓解压力//只修改了a，后续可能需要修改其他东西
                steering = np.clip(self.steering_control(self.target_lane_index), -self.MAX_STEERING_ANGLE,
                                   self.MAX_STEERING_ANGLE)
                ego_vehicle.sub_step(1 / self.mpc_dt, u[t], steering)
                # 初值
                if i == 0:
                    opti.set_initial(u[t], self.KP_A * (target_speed - self.speed))
                else:
                    opti.set_initial(u[t], -i)  # 如果原先的问题无解，多半是碰撞了，所以从减速那找可能的解
                objective += (ego_vehicle.speed - target_speed) ** 2  # 每次循环添加当前变量与目标值差的平方，构成目标函数，可以考虑加上车道中心线的距离
                # 添加变量的约束条件

                if isinstance(ego_vehicle.position, ca.MX):
                    for vehicles in observation_vehicle:
                        a = vehicles.position - ego_vehicle.position
                        opti.subject_to(a.T @ a > self.min_distance ** 2)  # 这里加以限制，可能是场之类的，目前先是欧式距离

                opti.subject_to(opti.bounded(-self.MAX_A, u[t], self.MAX_A))
            opti.minimize(objective)
            # 选择求解器并求解
            opts = {
                'verbose': False,  # 关闭求解器的详细输出
                'ipopt': {
                    'print_level': 0,  # 设置 IPOPT 的打印级别为 0
                    'sb': 'yes'  # 关闭 IPOPT 的启动横幅信息
                }
            }
            opti.solver('ipopt', opts)  # 使用 IPOPT 求解器
            try:
                # 保存原始的 stdout 和 stderr
                original_stdout = sys.stdout
                original_stderr = sys.stderr
                # 设置虚拟的输出流
                sys.stdout = DummyFile()
                sys.stderr = DummyFile()
                solution = opti.solve()
                a = solution.value(u)[0]
                success = True
                break
            except RuntimeError:
                success = False
            finally:
                # 恢复原始的 stdout 和 stderr
                sys.stdout = original_stdout
                sys.stderr = original_stderr

        if not success:
            # print("an inevitable collision")
            a = self.KP_A * (target_speed - self.speed)
        # print(success)
        return a




    def get_routes_at_intersection(self) -> List[Route]:
        """Get the list of routes that can be followed at the next intersection."""
        if not self.route:
            return []
        for index in range(min(len(self.route), 3)):
            try:
                next_destinations = self.road.network.graph[self.route[index][1]]
            except KeyError:
                continue
            if len(next_destinations) >= 2:
                break
        else:
            return [self.route]
        next_destinations_from = list(next_destinations.keys())
        routes = [
            self.route[0 : index + 1]
            + [(self.route[index][1], destination, self.route[index][2])]
            for destination in next_destinations_from
        ]
        return routes

    def set_route_at_intersection(self, _to: int) -> None:
        """
        Set the road to be followed at the next intersection.

        Erase current planned route.

        :param _to: index of the road to follow at next intersection, in the road network
        """

        routes = self.get_routes_at_intersection()
        if routes:
            if _to == "random":
                _to = self.road.np_random.integers(len(routes))
            self.route = routes[_to % len(routes)]

    def predict_trajectory_constant_speed(
        self, times: np.ndarray
    ) -> Tuple[List[np.ndarray], List[float]]:
        """
        Predict the future positions of the vehicle along its planned route, under constant speed

        :param times: timesteps of prediction
        :return: positions, headings
        """
        coordinates = self.lane.local_coordinates(self.position)
        route = self.route or [self.lane_index]
        pos_heads = [
            self.road.network.position_heading_along_route(
                route, coordinates[0] + self.speed * t, 0, self.lane_index
            )
            for t in times
        ]
        return tuple(zip(*pos_heads))


class MDPVehicle(ControlledVehicle):
    # myrl
    SPEED_COUNT: int = 5  # [], original = 3
    SPEED_MIN: float = 10  # [m/s]
    SPEED_MAX: float = 30  # [m/s]
    """A controlled vehicle with a specified discrete range of allowed target speeds."""

    DEFAULT_TARGET_SPEEDS = np.linspace(10, 30, 3)#(10, 20, 30)

    def __init__(
        self,
        road: Road,
        position: List[float],
        heading: float = 0,
        speed: float = 0,
        target_lane_index: Optional[LaneIndex] = None,
        target_speed: Optional[float] = None,
        target_speeds: Optional[Vector] = None,
        route: Optional[Route] = None,
        # mychange
        low_controller: str = "MPC",
    ) -> None:
        """
        Initializes an MDPVehicle

        :param road: the road on which the vehicle is driving
        :param position: its position
        :param heading: its heading angle
        :param speed: its speed
        :param target_lane_index: the index of the lane it is following
        :param target_speed: the speed it is tracking
        :param target_speeds: the discrete list of speeds the vehicle is able to track, through faster/slower actions
        :param route: the planned route of the vehicle, to handle intersections
        :param route: low_controller: PID
        """
        super().__init__(
            road,
            position,
            heading,
            speed,
            target_lane_index,
            target_speed, route,

        )
        self.target_speeds = (
            np.array(target_speeds)
            if target_speeds is not None
            else self.DEFAULT_TARGET_SPEEDS
        )
        self.speed_index = self.speed_to_index(self.target_speed)
        self.target_speed = self.index_to_speed(self.speed_index)
        # mychange
        self.low_controller = low_controller
        # print("self.low_controller_INIT==", self.low_controller)

    def act(self, action: Union[dict, str] = None) -> None:
        """
        Perform a high-level action.

        - If the action is a speed change, choose speed from the allowed discrete range.
        - Else, forward action to the ControlledVehicle handler.

        :param action: a high-level action
        """
        # for vehicle in self.controlled_vehicles:
        #     # print(vehicle)
        #     vehicle.plan_route_to(vehicle, "nxr1", 0)
        # # TODO PLAN AGAIN
        # if self.re_plan:


        if action == "FASTER":
            self.speed_index = self.speed_to_index(self.speed) + 1
        elif action == "SLOWER":
            self.speed_index = self.speed_to_index(self.speed) - 1
        else:
            super().act(action)
            return
        self.speed_index = int(np.clip(self.speed_index, 0, self.SPEED_COUNT - 1))
        self.target_speed = self.index_to_speed(self.speed_index)
        super().act()



    def index_to_speed(self, index: int) -> float:
        """
        Convert an index among allowed speeds to its corresponding speed

        :param index: the speed index []
        :return: the corresponding speed [m/s]
        """
        if self.SPEED_COUNT > 1:
            return self.SPEED_MIN + index * (self.SPEED_MAX - self.SPEED_MIN) / (self.SPEED_COUNT - 1)
        else:
            return self.SPEED_MIN

    # myrl 返回与速度接近的索引
    def speed_to_index(self, speed: float) -> int:
        """
        Find the index of the closest speed allowed to a given speed.

        :param speed: an input speed [m/s]
        :return: the index of the closest speed allowed []
        """
        x = (speed - self.SPEED_MIN) / (self.SPEED_MAX - self.SPEED_MIN)
        return np.int(np.clip(np.round(x * (self.SPEED_COUNT - 1)), 0, self.SPEED_COUNT - 1))



    @classmethod


    def speed_to_index_default(cls, speed: float) -> int:
        """
        Find the index of the closest speed allowed to a given speed.

        :param speed: an input speed [m/s]
        :return: the index of the closest speed allowed []
        """
        x = (speed - cls.SPEED_MIN) / (cls.SPEED_MAX - cls.SPEED_MIN)
        return np.int(np.clip(np.round(x * (cls.SPEED_COUNT - 1)), 0, cls.SPEED_COUNT - 1))


    @classmethod
    def get_speed_index(cls, vehicle: Vehicle) -> int:
        return getattr(
            vehicle, "speed_index", cls.speed_to_index_default(vehicle.speed)
        )

    def predict_trajectory(
        self,
        actions: List,
        action_duration: float,
        trajectory_timestep: float,
        dt: float,
    ) -> List[ControlledVehicle]:
        """
        Predict the future trajectory of the vehicle given a sequence of actions.

        :param actions: a sequence of future actions.
        :param action_duration: the duration of each action.
        :param trajectory_timestep: the duration between each save of the vehicle state.
        :param dt: the timestep of the simulation
        :return: the sequence of future states
        """
        states = []
        v = copy.deepcopy(self)
        t = 0
        for action in actions:
            v.act(action)  # High-level decision
            for _ in range(int(action_duration / dt)):
                t += 1
                v.act()  # Low-level control action
                v.step(dt)
                if (t % int(trajectory_timestep / dt)) == 0:
                    states.append(copy.deepcopy(v))
        return states
