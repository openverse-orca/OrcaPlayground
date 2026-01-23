from .utils.robot import Robot
from threading import Semaphore
import numpy as np

class LowState:

    class MotorState:
        def __init__(self):
            self.q = 0.0
            self.dq = 0.0
            self.ddq = 0.0
            self.tau_est = 0.0

    class ImuState:
        def __init__(self):
            self.quaternion = [1.0, 0.0, 0.0, 0.0]
            self.gyroscope = [0.0, 0.0, 0.0]

    def __init__(self):
        self.ndim = 29
        self.motor_state = [self.MotorState() for _ in range(self.ndim)]
        self.imu_state = self.ImuState()
        self.tick = 0

class StateProcessor:
    def __init__(self, config, low_state: LowState):
        self.config = config
        self.robot = Robot(config)
        self.num_dof = self.robot.NUM_JOINTS
        self._init_q = np.zeros(3 + 4 + self.num_dof)
        self.q = self._init_q
        self.dq = np.zeros(3 + 3 + self.num_dof)
        self.tau_est = np.zeros(self.num_dof)
        self.temp_first = np.zeros(self.num_dof)
        self.temp_second = np.zeros(self.num_dof)
        self.low_state : LowState = low_state

    def _prepare_low_state(self):
        imu_state = self.low_state.imu_state
        self.q[0:3] = 0.0
        self.q[3:7] = imu_state.quaternion # w, x, y, z
        self.dq[3:6] = imu_state.gyroscope
        joint_state = self.low_state.motor_state
        for i in range(self.num_dof):
            self.q[7+i] = joint_state[self.robot.JOINT2MOTOR[i]].q
            self.dq[6+i] = joint_state[self.robot.JOINT2MOTOR[i]].dq
        robot_state_data = np.array(self.q.tolist() + self.dq.tolist(), dtype=np.float64).reshape(1, -1)
        return robot_state_data

class LowCommand:
    class MotorCommand:
        def __init__(self):
            self.q = 0.0
            self.dq = 0.0
            self.kp = 0.0
            self.kd = 0.0
            self.tau = 0.0

    def __init__(self):
        self.ndim = 29
        self.motor_command = [self.MotorCommand() for _ in range(self.ndim)]


class CommandSender:
    def __init__(self, config, low_command: LowCommand):
        self.config = config
        self.robot = Robot(self.config)
        self.kp_level = 1.0 # 0.1
        self.waist_kp_level = 1.0
        self.robot_kp = np.zeros(self.robot.NUM_MOTORS)
        self.robot_kd = np.zeros(self.robot.NUM_MOTORS)
        # set kp level
        for i in range(len(self.robot.MOTOR_KP)):
            self.robot_kp[i] = self.robot.MOTOR_KP[i] * self.kp_level
        for i in range(len(self.robot.MOTOR_KD)):
            self.robot_kd[i] = self.robot.MOTOR_KD[i] * 1.0

        self.ndim = self.robot.NUM_MOTORS
        self.low_command : LowCommand = low_command
        self.init_low_command()
    
    def init_low_command(self):
        # 初始化为默认站立姿态，并设置正确的 kp/kd，防止启动时摔倒
        for i in range(self.robot.NUM_MOTORS):
            motor_index = self.robot.JOINT2MOTOR[i]
            joint_index = self.robot.MOTOR2JOINT[i]
            # 使用默认站立角度
            self.low_command.motor_command[motor_index].q = self.robot.DEFAULT_DOF_ANGLES[joint_index]
            self.low_command.motor_command[motor_index].dq = 0.0
            self.low_command.motor_command[motor_index].tau = 0.0
            # 设置正确的 kp/kd
            self.low_command.motor_command[motor_index].kp = self.robot_kp[motor_index]
            self.low_command.motor_command[motor_index].kd = self.robot_kd[motor_index]

    def update_command(self, cmd_q, cmd_dq, cmd_tau):
        for i in range(self.robot.NUM_MOTORS):
            motor_index = self.robot.JOINT2MOTOR[i]
            joint_index = self.robot.MOTOR2JOINT[i]
            # print(f"motor_index: {motor_index}, joint_index: {joint_index}")

            self.low_command.motor_command[motor_index].q = cmd_q[joint_index]
            self.low_command.motor_command[motor_index].dq = cmd_dq[joint_index]
            self.low_command.motor_command[motor_index].tau = cmd_tau[joint_index]
            # kp kd
            self.low_command.motor_command[motor_index].kp = self.robot_kp[motor_index]
            self.low_command.motor_command[motor_index].kd = self.robot_kd[motor_index]

class ShareState:
    def __init__(self):
        self.low_state = LowState()
        self.low_command = LowCommand()
        self.low_state_semaphore = Semaphore(1)
        self.low_command_semaphore = Semaphore(1)