from typing import Literal

import threading
import numpy as np
import time
import grpc
# from std_msgs.msg import Float32MultiArray

from scipy.spatial.transform import Rotation
# import torch
import pygame
# from pynput import keyboard
from sshkeyboard import listen_keyboard, stop_listening
from termcolor import colored
import onnxruntime
# import ipdb; ipdb.set_trace()
import sys
sys.path.append('../')
sys.path.append('./')

from ..utils.robot import Robot
from ..utils.history_handler import HistoryHandler

from ..share_state import ShareState, CommandSender, StateProcessor 
from orca_gym.log.orca_log import get_orca_logger, OrcaLog
from orca_gym.protos.mjc_message_pb2_grpc import GrpcServiceStub
import orca_gym.protos.mjc_message_pb2 as mjc_message_pb2

orca_logger = OrcaLog.get_instance()

KeyboardInputMode = Literal["orcastudio", "console"]


class SceneKeyboardInput:
    def __init__(self, grpc_address: str):
        self._channel = grpc.insecure_channel(grpc_address)
        self._stub = GrpcServiceStub(self._channel)
        self._request = mjc_message_pb2.GetKeyPressedEventsRequest()
        self.keyboard_state = {
            "W": 0, "A": 0, "S": 0, "D": 0,
            "Q": 0, "E": 0, "Z": 0, "O": 0, "I": 0, "R": 0,
            "Space": 0, "Up": 0, "Down": 0, "Left": 0, "Right": 0,
            "0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0, "9": 0,
            ",": 0, ".": 0,
        }
        self._event_map = {
            "keyboard_key_alphanumeric_W": "W",
            "keyboard_key_alphanumeric_A": "A",
            "keyboard_key_alphanumeric_S": "S",
            "keyboard_key_alphanumeric_D": "D",
            "keyboard_key_alphanumeric_Q": "Q",
            "keyboard_key_alphanumeric_E": "E",
            "keyboard_key_alphanumeric_Z": "Z",
            "keyboard_key_alphanumeric_O": "O",
            "keyboard_key_alphanumeric_I": "I",
            "keyboard_key_alphanumeric_R": "R",
            "keyboard_key_alphanumeric_0": "0",
            "keyboard_key_alphanumeric_1": "1",
            "keyboard_key_alphanumeric_2": "2",
            "keyboard_key_alphanumeric_3": "3",
            "keyboard_key_alphanumeric_4": "4",
            "keyboard_key_alphanumeric_5": "5",
            "keyboard_key_alphanumeric_6": "6",
            "keyboard_key_alphanumeric_7": "7",
            "keyboard_key_alphanumeric_8": "8",
            "keyboard_key_alphanumeric_9": "9",
            "keyboard_key_edit_space": "Space",
            "keyboard_key_navigation_arrow_up": "Up",
            "keyboard_key_navigation_arrow_down": "Down",
            "keyboard_key_navigation_arrow_left": "Left",
            "keyboard_key_navigation_arrow_right": "Right",
            "keyboard_key_punctuation_comma": ",",
            "keyboard_key_punctuation_period": ".",
        }

    def update(self):
        for key in self.keyboard_state:
            self.keyboard_state[key] = 0

        response = self._stub.GetKeyPressedEvents(self._request)
        for event in response.events:
            key_name = self._event_map.get(event)
            if key_name is not None:
                self.keyboard_state[key_name] = 1

    def get_state(self):
        return self.keyboard_state.copy()

    def close(self):
        self._channel.close()

class BasePolicy:
    def __init__(self, 
                 config, 
                 model_path, 
                 share_state: ShareState,
                 policy_action_scale=0.25, 
                 decimation=4,
                 orcagym_addr: str | None = None,
                 keyboard_input: KeyboardInputMode = "orcastudio"):
        self.config = config
        self.robot = Robot(config)
        self.robot_state_data = None
        
        self.share_state = share_state
        self.command_sender = CommandSender(config, share_state.low_command)
        self.state_processor = StateProcessor(config, share_state.low_state)
        share_state.low_command_semaphore.acquire()

        self.setup_policy(model_path)

        self.num_dofs = self.robot.NUM_JOINTS
        self.last_policy_action = np.zeros((1, self.num_dofs))
        self.default_dof_angles = np.array(self.robot.DEFAULT_DOF_ANGLES).reshape(1, -1)
        self.policy_action_scale = policy_action_scale

        # Keypress control state
        self.use_policy_action = False

        self.last_time = time.time()

        self.decimation = decimation

        self.first_time_init = True
        self.init_count = 0
        self.get_ready_state = False

        self.lin_vel_command = np.array([[0., 0.]])
        self.ang_vel_command = np.array([[0.]])
        self.stand_command = np.array([[0]])
        self.default_base_height_command = np.array([[0.78]])
        self.base_height_command = self.default_base_height_command.copy()
        
        self.motor_pos_lower_limit_list = self.config.get("motor_pos_lower_limit_list", None)
        self.motor_pos_upper_limit_list = self.config.get("motor_pos_upper_limit_list", None)
        self.motor_vel_limit_list = self.config.get("motor_vel_limit_list", None)
        self.motor_effort_limit_list = self.config.get("motor_effort_limit_list", None)
        
        self.use_history = self.config["USE_HISTORY"]
        self.obs_scales = self.config["obs_scales"]
        self.history_handler = None
        self.current_obs = None
        if self.use_history: 
            self.history_handler = HistoryHandler(self.config["history_config"], self.config["obs_dims"])
            self.current_obs = {key: np.zeros((1, self.config["obs_dims"][key])) for key in self.config["obs_dims"].keys()}

        self._keyboard = None
        self._last_key_state = {}
        if keyboard_input == "orcastudio":
            if not orcagym_addr:
                raise ValueError(
                    "keyboard_input='orcastudio' requires a non-empty orcagym_addr for scene keyboard gRPC."
                )
            self._keyboard = SceneKeyboardInput(orcagym_addr)
            self._last_key_state = self._keyboard.get_state()
        elif keyboard_input != "console":
            raise ValueError(
                f"Unknown keyboard_input: {keyboard_input!r}; use 'orcastudio' or 'console'."
            )

        self._shutdown_event = threading.Event()
        self.key_listener_thread = None
        if self._keyboard is None:
            self.key_listener_thread = threading.Thread(target=self.start_key_listener)
            self.key_listener_thread.start()

    def setup_policy(self, model_path):
        # load onnx policy
        self.onnx_policy_session = onnxruntime.InferenceSession(model_path)
        self.onnx_input_name = self.onnx_policy_session.get_inputs()[0].name
        self.onnx_output_name = self.onnx_policy_session.get_outputs()[0].name
        def policy_act(obs):
            return self.onnx_policy_session.run([self.onnx_output_name], {self.onnx_input_name: obs})[0]
        self.policy = policy_act

    def prepare_obs_for_rl(self, robot_state_data):
        # robot_state_data [:3]: robot base pos
        # robot_state_data [3:7]: robot base quaternion
        # robot_state_data [7:7+dof_num]: joint angles 
        # robot_state_data [7+dof_num: 7+dof_num+3]: base linear velocity
        # robot_state_data [7+dof_num+3: 7+dof_num+6]: base angular velocity
        # robot_state_data [7+dof_num+6: 7+dof_num+6+dof_num]: joint velocities
        raise NotImplementedError


    def get_init_target(self, robot_state_data):
        dof_pos = robot_state_data[:, 7:7+self.num_dofs]
        if self.get_ready_state:
            # interpolate from current dof_pos to default angles
            q_target = dof_pos + (self.default_dof_angles - dof_pos) * (self.init_count / 500)
            self.init_count += 1
            return q_target
        else:
            return dof_pos

    def _get_obs_history(self,):
        assert "history_config" in self.config.keys()
        history_config = self.config["history_config"]
        history_list = []
        for key in sorted(history_config.keys()):
            history_length = history_config[key]
            history_array = self.history_handler.query(key)[:, :history_length]
            history_array = history_array.reshape(history_array.shape[0], -1)  # Shape: [4096, history_length*obs_dim]
            history_list.append(history_array)
        return np.concatenate(history_list, axis=1)
    
    def get_policy_action(self, robot_state_data):
        # Process low states
        obs = self.prepare_obs_for_rl(robot_state_data)
        # Policy inference
        policy_action = self.policy(obs)
        policy_action = np.clip(policy_action, -100, 100)

        self.last_policy_action = policy_action.copy()  
        scaled_policy_action = policy_action * self.policy_action_scale

        return scaled_policy_action

    def rl_inference(self):
        # 等待主线程更新状态（主线程在 step 结束时释放 low_state_semaphore）
        self.share_state.low_state_semaphore.acquire()
        
        try:
            if self._shutdown_event.is_set():
                return

            self.robot_state_data = self.state_processor._prepare_low_state()
            if self.robot_state_data is None:
                print("No robot state data received, skipping rl inference")
                return
            
            # Get policy action
            scaled_policy_action = self.get_policy_action(self.robot_state_data)
            if self.get_ready_state:
                q_target = self.get_init_target(self.robot_state_data)
                if self.init_count >= 500:
                    self.init_count = 500
                    # 插值完成后自动切换到策略控制
                    self.get_ready_state = False
                    self.use_policy_action = True
                    orca_logger.info("Init done, switching to policy control")
            elif not self.use_policy_action:
                q_target = self.robot_state_data[:, 7:7+self.num_dofs]
            else:
                if not scaled_policy_action.shape[1] == self.num_dofs:
                    scaled_policy_action = np.concatenate([scaled_policy_action, np.zeros((1, self.num_dofs - scaled_policy_action.shape[1]))], axis=1)
                q_target = scaled_policy_action + self.default_dof_angles

            # Clip q target
            if self.motor_pos_lower_limit_list and self.motor_pos_upper_limit_list:
                q_target[0] = np.clip(q_target[0], self.motor_pos_lower_limit_list, self.motor_pos_upper_limit_list)

            # Send command
            cmd_q = q_target[0]
            cmd_dq = np.zeros(self.num_dofs)
            cmd_tau = np.zeros(self.num_dofs)
            self.command_sender.update_command(cmd_q, cmd_dq, cmd_tau)
        finally:
            # 无论成功与否，都要释放 low_command_semaphore，允许主线程继续
            self.share_state.low_command_semaphore.release()

    def start_key_listener(self):
        """Start a key listener using pynput."""
        def on_press(keycode):
            try:
                self.handle_keyboard_button(keycode)
            except AttributeError:
                pass  # Handle special keys if needed

        try:
            listen_keyboard(on_press=on_press, until=None)
        except Exception as exc:
            if not self._shutdown_event.is_set():
                orca_logger.warning(f"Keyboard listener exited unexpectedly: {exc}")
    def _poll_scene_keyboard(self):
        if self._keyboard is None:
            return

        self._keyboard.update()
        key_state = self._keyboard.get_state()

        edge_key_map = {
            "W": "w",
            "A": "a",
            "S": "s",
            "D": "d",
            "Q": "q",
            "E": "e",
            "Z": "z",
            "O": "o",
            "I": "i",
            "R": "r",
            "0": "0",
            "1": "1",
            "2": "2",
            "3": "3",
            "4": "4",
            "5": "5",
            "6": "6",
            "7": "7",
            "8": "8",
            "9": "9",
            ",": ",",
            ".": ".",
        }

        for source_key, translated_key in edge_key_map.items():
            if self._last_key_state.get(source_key, 0) == 0 and key_state.get(source_key, 0) == 1:
                self.handle_keyboard_button(translated_key)

        self._last_key_state = key_state

    def _on_sim_reset_requested(self):
        pass

    def _request_sim_reset(self):
        self.use_policy_action = True
        self.get_ready_state = False
        self.init_count = 0
        self.lin_vel_command[:] = 0.0
        self.ang_vel_command[:] = 0.0
        self.stand_command[:] = 0
        self.base_height_command = self.default_base_height_command.copy()
        self.last_policy_action[:] = 0.0
        if self.history_handler is not None:
            self.history_handler.reset([0])
        self.command_sender.reset_gains()
        self.command_sender.init_low_command()
        self._on_sim_reset_requested()
        self.share_state.reset_requested = True
        orca_logger.info("已请求仿真重置")

    def _handle_keyboard_button_impl(self, keycode):
        """处理键盘事件的实现，子类可以重写此方法扩展功能

        主键位使用数字键 1–5（原 F1–F5），便于 sshkeyboard 等终端输入；高度与 Kp 使用 `,` `.` 与 6–0。
        """
        if keycode == "1":
            self.use_policy_action = True
            self.get_ready_state = False
            self.phase = 0.0
        elif keycode == "r":
            self._request_sim_reset()
        elif keycode == "o":
            self.use_policy_action = False
            self.get_ready_state = False
            orca_logger.info("Actions set to zero")
        elif keycode == "i":
            self.get_ready_state = True
            self.init_count = 0
            orca_logger.info("Setting to init state")
        elif keycode == "w" and self.stand_command[0, 0]:
            self.lin_vel_command[0, 0]+=0.1
        elif keycode == "s" and self.stand_command[0, 0]:
            self.lin_vel_command[0, 0]-=0.1
        elif keycode == "a" and self.stand_command[0, 0]:
            self.lin_vel_command[0, 1]+=0.1 
        elif keycode == "d" and self.stand_command[0, 0]:
            self.lin_vel_command[0, 1]-=0.1
        elif keycode == "q":
            self.ang_vel_command[0, 0]-=0.1
        elif keycode == "e":
            self.ang_vel_command[0, 0]+=0.1
        elif keycode == "z":
            self.ang_vel_command[0, 0] = 0.
            self.lin_vel_command[0, 0] = 0.
            self.lin_vel_command[0, 1] = 0.
        elif keycode == ",":
            self.base_height_command += 0.05
        elif keycode == ".":
            self.base_height_command -= 0.05
        elif keycode == "6":
            self.command_sender.set_kp_level(self.command_sender.kp_level - 0.1)
        elif keycode == "7":
            self.command_sender.set_kp_level(self.command_sender.kp_level - 0.01)
        elif keycode == "8":
            self.command_sender.set_kp_level(self.command_sender.kp_level + 0.01)
        elif keycode == "9":
            self.command_sender.set_kp_level(self.command_sender.kp_level + 0.1)
        elif keycode == "0":
            self.command_sender.reset_gains()
        elif keycode == "2":
            self.stand_command = 1 - self.stand_command
            if self.stand_command[0, 0] == 0:
                self.ang_vel_command[0, 0] = 0.
                self.lin_vel_command[0, 0] = 0.
                self.lin_vel_command[0, 1] = 0.

    def handle_keyboard_button(self, keycode):
        """处理键盘事件，带信号量保护"""
        self.share_state.low_state_semaphore.acquire()
        try:
            self._handle_keyboard_button_impl(keycode)
            print(f"Linear velocity command: {self.lin_vel_command}")
            print(f"Angular velocity command: {self.ang_vel_command}")
            print(f"Base height command: {self.base_height_command}")
            print(f"Stand command: {self.stand_command}")
        finally:
            self.share_state.low_state_semaphore.release()

    def odometry_callback(self, msg):
        # Extract current position from odometry
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z
        self.mocap_pos = np.array([x, y, z])
        
        # Convert orientation from quaternion to Euler angles
        quat = msg.pose.pose.orientation
        self.mocap_quat = np.array([quat.x, quat.y, quat.z, quat.w])
        rot = Rotation.from_quat(self.mocap_quat)
        self.mocap_euler = rot.as_euler('xyz')
        twist = msg.twist.twist
        lin_vel = np.array([twist.linear.x, twist.linear.y, twist.linear.z])
        lin_vel = rot.inv().apply(lin_vel)
        self.mocap_lin_vel = lin_vel
        # print(f"Current position: {self.mocap_pos}, Current orientation: {self.mocap_euler}")

    def run(self):
        total_inference_cnt = 0
        start_time = time.time()
        try:
            while not self._shutdown_event.is_set():
                # if self.use_joystick and self.wc_msg is not None:
                #     self.process_joystick_input()
                self._poll_scene_keyboard()
                self.rl_inference()
                end_time = time.time()
                total_inference_cnt += 1

        except KeyboardInterrupt:
            pass

    def close(self, join_timeout=2.0):
        """Stop background workers so stdin/terminal state can be restored."""
        if self._shutdown_event.is_set():
            return

        self._shutdown_event.set()

        try:
            stop_listening()
        except Exception:
            pass

        if self._keyboard is not None:
            try:
                self._keyboard.close()
            except Exception:
                pass

        # Wake up any waiting worker so run() can observe the shutdown flag.
        self.share_state.low_state_semaphore.release()
        self.share_state.low_command_semaphore.release()

        if self.key_listener_thread is not None and self.key_listener_thread.is_alive():
            self.key_listener_thread.join(timeout=join_timeout)