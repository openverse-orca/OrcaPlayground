
import numpy as np


from .robot import Robot

class StateProcessor:
    def __init__(self, config):
        self.config = config
        self.robot = Robot(config)

        self.num_dof = self.robot.NUM_JOINTS
        # 3 + 4 + 19
        self._init_q = np.zeros(3 + 4 + self.num_dof)
        self.q = self._init_q
        self.dq = np.zeros(3 + 3 + self.num_dof)
        self.tau_est = np.zeros(self.num_dof)
        self.temp_first = np.zeros(self.num_dof)
        self.temp_second = np.zeros(self.num_dof)
        self.robot_low_state = None

    def _prepare_low_state(self):
        if not self.robot_low_state:
            print("No low state received")
            return
        imu_state = self.robot_low_state.imu_state
        # base quaternion
        self.q[0:3] = 0.0
        self.q[3:7] = imu_state.quaternion # w, x, y, z
        self.dq[3:6] = imu_state.gyroscope
        unitree_joint_state = self.robot_low_state.motor_state

        for i in range(self.num_dof):
            # import ipdb; ipdb.set_trace()
            self.q[7+i] = unitree_joint_state[self.robot.JOINT2MOTOR[i]].q
            self.dq[6+i] = unitree_joint_state[self.robot.JOINT2MOTOR[i]].dq
        
        robot_state_data = np.array(self.q.tolist() + self.dq.tolist(), dtype=np.float64).reshape(1, -1)

        return robot_state_data
