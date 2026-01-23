
import numpy as np
import time
import threading
# from std_msgs.msg import Float32MultiArray

from scipy.spatial.transform import Rotation
# import torch
import pygame
# from pynput import keyboard
from sshkeyboard import listen_keyboard
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

orca_logger = OrcaLog.get_instance()

class BasePolicy:
    def __init__(self, 
                 config, 
                 model_path, 
                 share_state: ShareState,
                 policy_action_scale=0.25, 
                 decimation=4):
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
        self.base_height_command = np.array([[0.78]])
        
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


        self.key_listener_thread = threading.Thread(target=self.start_key_listener, daemon=True)
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

        listener = listen_keyboard(on_press=on_press)
        listener.start()
        listener.join()  # Keep the thread alive

    def _handle_keyboard_button_impl(self, keycode):
        """处理键盘事件的实现，子类可以重写此方法扩展功能"""
        if keycode == "]":
            self.use_policy_action = True
            self.get_ready_state = False
            self.phase = 0.0
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
        elif keycode == "1":
            self.base_height_command += 0.05
        elif keycode == "2":
            self.base_height_command -= 0.05
        elif keycode == "5":
            self.command_sender.kp_level -= 0.01
            for i in range(len(self.command_sender.robot_kp)):
                self.command_sender.robot_kp[i] = self.robot.MOTOR_KP[i] * self.command_sender.kp_level
        elif keycode == "6":
            self.command_sender.kp_level += 0.01
            for i in range(len(self.command_sender.robot_kp)):
                self.command_sender.robot_kp[i] = self.robot.MOTOR_KP[i] * self.command_sender.kp_level
        elif keycode == "4":
            self.command_sender.kp_level -= 0.1
            for i in range(len(self.command_sender.robot_kp)):
                self.command_sender.robot_kp[i] = self.robot.MOTOR_KP[i] * self.command_sender.kp_level
        elif keycode == "7":
            self.command_sender.kp_level += 0.1
            for i in range(len(self.command_sender.robot_kp)):
                self.command_sender.robot_kp[i] = self.robot.MOTOR_KP[i] * self.command_sender.kp_level
        elif keycode == "0":
            self.command_sender.kp_level = 1.0
            for i in range(len(self.command_sender.robot_kp)):
                self.command_sender.robot_kp[i] = self.robot.MOTOR_KP[i] * self.command_sender.kp_level
        elif keycode == "=":
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
            while True:
                # if self.use_joystick and self.wc_msg is not None:
                #     self.process_joystick_input()
                self.rl_inference()
                end_time = time.time()
                total_inference_cnt += 1

        except KeyboardInterrupt:
            pass