"""Euler task environment for the complete sensor demonstration scenes."""

from pathlib import Path
import xml.etree.ElementTree as ET

from gymnasium import spaces
import numpy as np

from orca_gym.environment.euler.orca_gym_euler_env import OrcaGymEulerEnv


class ProviderSceneEnv(OrcaGymEulerEnv):
    """Advance one simulation and expose copied vendor outputs as observations."""

    def __init__(self, xml_path, provider_manifests, *, sensor_shapes,
                 frame_skip=1, host_path=None, initial_keyframe=None):
        root = ET.parse(xml_path).getroot()
        option = root.find('option')
        timestep = float(option.get('timestep', '0.002')) if option is not None else .002
        self._sensor_shapes = dict(sensor_shapes)
        self._has_provider_sample = False
        super().__init__(
            frame_skip=frame_skip, orcagym_addr='localhost:50051', agent_names=[],
            time_step=timestep, model_xml_path=str(Path(xml_path).resolve()),
            skip_grpc_load=True, render_mode='none',
            sensor_provider_manifests=[str(path) for path in provider_manifests],
            sensor_host_path=str(host_path) if host_path is not None else None,
        )
        try:
            self._reset_qpos = self.init_qpos.copy()
            self._reset_qvel = self.init_qvel.copy()
            self._reset_ctrl = np.zeros(self.model.nu, dtype=np.float64)
            if initial_keyframe is not None:
                keys = [key for key in root.findall('keyframe/key')
                        if key.get('name') == initial_keyframe]
                if len(keys) != 1:
                    raise ValueError(f'Expected one exact initial keyframe: {initial_keyframe}')
                for name, destination in (('qpos', self._reset_qpos),
                                          ('qvel', self._reset_qvel),
                                          ('ctrl', self._reset_ctrl)):
                    if name in keys[0].attrib:
                        values = np.asarray(keys[0].get(name).split(), dtype=np.float64)
                        if values.shape != destination.shape or not np.isfinite(values).all():
                            raise ValueError(f'Initial keyframe {name} has an invalid shape or value')
                        destination[:] = values
            lower = np.full(self.model.nu, -np.inf, dtype=np.float64)
            upper = np.full(self.model.nu, np.inf, dtype=np.float64)
            for actuator in self.model.get_actuator_dict().values():
                if actuator['CtrlLimited']:
                    lower[actuator['ActuatorId']], upper[actuator['ActuatorId']] = actuator['CtrlRange']
            self.action_space = spaces.Box(lower, upper, dtype=np.float64)
            self.observation_space = spaces.Dict({
                'provider_sensors': spaces.Dict({
                    name: spaces.Box(-np.inf, np.inf, shape=shape, dtype=np.float64)
                    for name, shape in self._sensor_shapes.items()
                }),
                'provider_valid': spaces.MultiBinary(1),
            })
        except BaseException:
            self.close()
            raise

    def reset_model(self):
        self._has_provider_sample = False
        self.set_joint_qpos(self._reset_qpos.copy())
        # The next setter reads the other state vector from DataView.
        self._sync_view()
        self.set_joint_qvel(self._reset_qvel.copy())
        self.set_ctrl(self._reset_ctrl.copy())
        self.mj_forward()
        self._sync_view()
        return self._get_obs(), {'provider_ready': False, 'simulation_time': float(self.data.time)}

    def _get_obs(self):
        if self._has_provider_sample:
            values = self.query_provider_sensor_data()
            values = {name: values[name] for name in self._sensor_shapes}
        else:
            values = {name: np.zeros(shape, dtype=np.float64)
                      for name, shape in self._sensor_shapes.items()}
        return {'provider_sensors': values,
                'provider_valid': np.array([self._has_provider_sample], dtype=np.int8)}

    def step(self, action):
        control = np.asarray(action, dtype=np.float64)
        if not np.isfinite(control).all() or not self.action_space.contains(control):
            raise ValueError('action must be a finite actuator vector within the declared control limits')
        self._has_provider_sample = False
        self.do_simulation(control, self.frame_skip)
        self._has_provider_sample = True
        return (self._get_obs(), 0.0, False, False,
                {'provider_ready': True, 'simulation_time': float(self.data.time)})
