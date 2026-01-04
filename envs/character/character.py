from orca_gym.environment.orca_gym_local_env import OrcaGymLocalEnv
import yaml
from orca_gym.devices.keyboard import KeyboardInput, KeyboardInputSourceType
import os
import numpy as np
import time
import orca_gym.utils.rotations as rotations

from orca_gym.log.orca_log import get_orca_logger
_logger = get_orca_logger()


class Character():
    def __init__(self, 
        env: OrcaGymLocalEnv, 
        agent_name: str,
        agent_id: int,
        character_name: str,
    ):
        self._env = env
        self._agent_name = agent_name
        self._agent_id = agent_id
        self._model = env.model
        self._data = env.data
        self._load_config(character_name)
        self._init_character()  
        self._keyboard = KeyboardInput(KeyboardInputSourceType.ORCASTUDIO)

    def _load_config(self, character_name: str):
        path = os.path.dirname(os.path.abspath(__file__))
        with open(f"{path}/character_config/{character_name}.yaml", 'r') as f:
            self._config = yaml.load(f, Loader=yaml.FullLoader)
            
    def _init_character(self):
        self._asset_path = self._config['asset_path']
        self._body_name = self._env.body(self._config['body_name'], self._agent_id)
        
        # 尝试查找body，如果失败则尝试其他可能的名称
        try:
            self._body_id = self._model.body_name2id(self._body_name)
        except KeyError:
            available_bodies = self._model.get_body_names()
            _logger.warning(f"找不到body: '{self._body_name}'")
            _logger.info(f"可用的body名称: {available_bodies}")
            
            # 尝试查找包含"Animation"或"animation"的body
            possible_names = [
                self._body_name,  # 原始名称
                self._config['body_name'],  # 不带agent前缀的名称
                f"{self._agent_name}_{self._config['body_name']}",  # 显式组合
            ]
            
            # 查找包含"Animation"的body
            for body in available_bodies:
                if "animation" in body.lower() or "Animation" in body:
                    possible_names.append(body)
            
            # 尝试每个可能的名称
            found = False
            for name in possible_names:
                try:
                    self._body_id = self._model.body_name2id(name)
                    self._body_name = name
                    _logger.info(f"成功找到body: '{name}' (使用备用名称)")
                    found = True
                    break
                except KeyError:
                    continue
            
            if not found:
                _logger.error(f"Agent名称: {self._agent_name}, Body名称配置: {self._config['body_name']}")
                _logger.error(f"尝试过的名称: {possible_names}")
                raise KeyError(f"找不到body: '{self._body_name}'. 可用的body: {available_bodies}")
        joint_name_dict = self._config['joint_names']
        self._joint_names = {
            "Move_X" : self._env.joint(joint_name_dict['Move_X'], self._agent_id),
            "Move_Y" : self._env.joint(joint_name_dict['Move_Y'], self._agent_id),
            "Move_Z" : self._env.joint(joint_name_dict['Move_Z'], self._agent_id),
            "Rotate_Z" : self._env.joint(joint_name_dict['Rotate_Z'], self._agent_id),
        }
        
        # 查找joint IDs，如果失败则尝试其他可能的名称
        self._joint_ids = {}
        available_joints = list(self._model.get_joint_dict().keys())
        
        for key, joint_name in self._joint_names.items():
            try:
                self._joint_ids[key] = self._model.joint_name2id(joint_name)
            except KeyError:
                _logger.warning(f"找不到joint: '{joint_name}' (用于 {key})")
                
                # 尝试查找包含配置中joint名称的joint
                config_joint_name = joint_name_dict[key]
                possible_names = [
                    joint_name,  # 原始名称
                    config_joint_name,  # 不带agent前缀的名称
                    f"{self._agent_name}_{config_joint_name}",  # 显式组合
                ]
                
                # 查找包含配置joint名称的joint
                for available_joint in available_joints:
                    if config_joint_name.lower() in available_joint.lower():
                        possible_names.append(available_joint)
                
                # 尝试每个可能的名称
                found = False
                for name in possible_names:
                    try:
                        self._joint_ids[key] = self._model.joint_name2id(name)
                        self._joint_names[key] = name
                        _logger.info(f"成功找到joint {key}: '{name}' (使用备用名称)")
                        found = True
                        break
                    except KeyError:
                        continue
                
                if not found:
                    _logger.error(f"找不到joint {key}: '{joint_name}'")
                    _logger.error(f"可用的joint名称: {available_joints}")
                    _logger.error(f"尝试过的名称: {possible_names}")
                    raise KeyError(f"找不到joint {key}: '{joint_name}'. 可用的joint: {available_joints}")
        
        self._ctrl_joint_qvel = {
            self._joint_names["Move_X"] : [0],
            self._joint_names["Move_Y"] : [0],
            self._joint_names["Rotate_Z"] : [0],
        }

        self._speed = self._config['speed']
        self._acceleration = self._speed['Acceleration']
        self._move_speed = 0.0
        self._turn_speed = 0.0

        self._control_type = self._config['control_type']

        self._keyboard_control = self._config['keyboard_control']
        self._waypoint_control = self._config['waypoint_control']
        self._waypoint_distance_threshold = self._config['waypoint_distance_threshold']
        self._waypoint_angle_threshold = np.deg2rad(self._config['waypoint_angle_threshold'])

        body_xpos, _, _ = self._env.get_body_xpos_xmat_xquat([self._body_name])
        self._original_coordinates = body_xpos[:2]
        _logger.info(f"Original Coordinates:  {self._original_coordinates}")

        
    def on_step(self):
        rotate_z_pos = self._env.query_joint_qpos([self._joint_names["Rotate_Z"]])[self._joint_names["Rotate_Z"]][0]
        heading = rotate_z_pos % (2 * np.pi)

        self._process_control_type_switch()

        if self._control_type['active_type'] == 'keyboard':
            self._process_keyboard_input(heading)
        elif self._control_type['active_type'] == 'waypoint':
            self._process_waypoint_input(heading)

        self._env.set_joint_qvel(self._ctrl_joint_qvel)

    def on_reset(self):
        self._reset_move_and_turn()
        ctrl_qpos = {
            self._joint_names["Rotate_Z"] : [0],
        }

        self._env.set_joint_qpos(ctrl_qpos)


    def _set_anim_param_bool(self, param_name: str, value: bool):
        """安全地设置动画参数，检查 scene_runtime 是否存在"""
        if hasattr(self._env, 'scene_runtime') and self._env.scene_runtime is not None:
            self._env.scene_runtime.set_actor_anim_param_bool(self._agent_name, param_name, value)
        else:
            _logger.warning(f"scene_runtime 未设置，无法设置动画参数 {param_name}={value}")

    def _turn_left(self):
        self._turn_speed = self._speed['TurnLeft']
        self._set_anim_param_bool("TurnLeft", True)
        self._set_anim_param_bool("TurnRight", False)

    def _turn_right(self):
        self._turn_speed = self._speed['TurnRight']
        self._set_anim_param_bool("TurnRight", True)
        self._set_anim_param_bool("TurnLeft", False)
    

    def _stop_turning(self):
        self._turn_speed = 0
        self._set_anim_param_bool("TurnLeft", False)
        self._set_anim_param_bool("TurnRight", False)

    def _move_forward(self):
        if self._move_speed < self._speed['Forward']:
            self._move_speed += self._speed['Forward'] * self._acceleration
        self._set_anim_param_bool("Forward", True)
        self._set_anim_param_bool("Backward", False)

    def _move_backward(self):
        if self._move_speed > self._speed['Backward']:
            self._move_speed += self._speed['Backward'] * self._acceleration
        self._set_anim_param_bool("Backward", True)
        self._set_anim_param_bool("Forward", False)

    def _stop_moving(self):
        self._move_speed = 0
        self._set_anim_param_bool("Forward", False)
        self._set_anim_param_bool("Backward", False)

    def _process_move(self, move_speed : float, heading : float):
        move_y_vel = move_speed * np.cos(heading)
        move_x_vel = move_speed * -np.sin(heading)

        self._ctrl_joint_qvel[self._joint_names["Move_X"]][0] = move_x_vel
        self._ctrl_joint_qvel[self._joint_names["Move_Y"]][0] = move_y_vel
        self._ctrl_joint_qvel[self._joint_names["Rotate_Z"]][0] = self._turn_speed

    def _reset_move_and_turn(self):
        self._move_speed = 0
        self._turn_speed = 0
        self._waypoint_time = 0
        self._waypoint_index = -1
        self._moving_to_waypoint = False
        self._next_waypoint_coord = None
        self._set_anim_param_bool("Forward", False)
        self._set_anim_param_bool("Backward", False)
        self._set_anim_param_bool("TurnLeft", False)
        self._set_anim_param_bool("TurnRight", False)


    def _process_control_type_switch(self):
        self._keyboard.update()
        keyboard_state = self._keyboard.get_state()
        # 安全地访问键盘状态，如果键不存在则默认为0
        waypoint_key = self._control_type['switch_key']['waypoint']
        keyboard_key = self._control_type['switch_key']['keyboard']
        
        if self._control_type['active_type'] != 'waypoint' and keyboard_state.get(waypoint_key, 0) == 1:
            self._control_type['active_type'] = 'waypoint'
            self._reset_move_and_turn()
            _logger.info("Switch to waypoint control")
        elif self._control_type['active_type'] != 'keyboard' and keyboard_state.get(keyboard_key, 0) == 1:
            self._control_type['active_type'] = 'keyboard'
            self._reset_move_and_turn()
            _logger.info("Switch to keyboard control")
        

    def _process_keyboard_input(self, heading : float):
        self._keyboard.update()
        keyboard_state = self._keyboard.get_state()

        # print("Speed: ", self._speed)

        # 安全地访问键盘状态，如果键不存在则默认为0
        if keyboard_state.get(self._keyboard_control['move_forward'], 0) == 1:
            self._move_forward()
        elif keyboard_state.get(self._keyboard_control['move_backward'], 0) == 1:
            self._move_backward()
        else:
            self._stop_moving()

        if keyboard_state.get(self._keyboard_control['turn_left'], 0) == 1:
            self._turn_left()
        elif keyboard_state.get(self._keyboard_control['turn_right'], 0) == 1:
            self._turn_right()
        else:
            self._stop_turning()

        # print("heading: ", heading, "move_speed: ", self._move_speed, "turn_speed: ", self._turn_speed)
        self._process_move(self._move_speed, heading)

    def _process_waypoint_input(self, heading : float):
        current_time = time.time()
        time_delta = current_time - self._waypoint_time

        if not self._moving_to_waypoint and time_delta > self._waypoint_control[self._waypoint_index]['Duration']:
            # time to go to next waypoint
            self._moving_to_waypoint = True
            self._waypoint_index += 1
            if self._waypoint_index >= len(self._waypoint_control):
                self._waypoint_index = 0
            
            next_waypoint_coord = np.array(self._waypoint_control[self._waypoint_index]['Coordinates']) + self._original_coordinates
            _logger.info(f"Next Waypoint Coordinates:  {next_waypoint_coord}")
            self._next_waypoint_coord = next_waypoint_coord

        if self._next_waypoint_coord is not None and self._moving_to_waypoint:
            body_xpos, _, body_xquat = self._env.get_body_xpos_xmat_xquat([self._body_name])
            current_coordinates = body_xpos[:2] 
            current_body_heading = rotations.quat2euler(body_xquat[:4])[2] % (2 * np.pi)

            # calculate the distance to the next waypoint, and the direction
            distance = np.linalg.norm(self._next_waypoint_coord - current_coordinates)
            direction = (self._next_waypoint_coord - current_coordinates) / distance
            # print("Distance: ", distance, "Direction: ", direction, "Body Xquat: ", body_xquat, "Current Body Heading: ", current_body_heading)
                
            # check if the character has reached the waypoint
            if distance < self._waypoint_distance_threshold:
                # reached the waypoint
                self._stop_moving()
                self._stop_turning()
                self._moving_to_waypoint = False
                self._waypoint_time = current_time
                self._process_move(self._move_speed, heading)
                return

            # check the direction of the character, if not facing the waypoint, turn to the waypoint
            # 计算目标方向角度
            target_angle = np.arctan2(direction[0], -direction[1])   
            # 计算角度差值（考虑圆周率）
            angle_diff = (target_angle - current_body_heading + np.pi) % (2*np.pi)
            if angle_diff > np.pi:
                angle_diff -= 2 * np.pi
            # print("Angle Error: ", angle_diff)
            if angle_diff < -self._waypoint_angle_threshold:
                self._turn_right()
            elif angle_diff > self._waypoint_angle_threshold:
                self._turn_left()
            else:
                self._move_forward()
                self._stop_turning()
                
            self._process_move(self._move_speed, heading)



            
