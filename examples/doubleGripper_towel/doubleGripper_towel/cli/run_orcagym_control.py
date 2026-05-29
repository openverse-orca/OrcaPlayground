"""Run auto-step control through OrcaGym (for OrcaStudio display mode)."""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
import sys
import time


def _register_env(gym, register, registry, env_name, orcagym_addr, frame_skip, time_step, agent_name, controller_min_steps):
    addr_tag = orcagym_addr.replace(':', '-')
    env_id = f'{env_name}-OrcaGym-{addr_tag}-{int(time.time())}'
    if env_id in registry:
        return env_id
    register(
        id=env_id,
        entry_point='doubleGripper_towel.envs.double_gripper_orcagym_env:DoubleGripperOrcaGymEnv',
        kwargs={
            'frame_skip': frame_skip,
            'orcagym_addr': orcagym_addr,
            'agent_names': [agent_name],
            'time_step': time_step,
            'controller_min_steps': controller_min_steps,
        },
        max_episode_steps=sys.maxsize,
    )
    return env_id


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Run double-gripper auto-step control via OrcaGym.'
    )
    parser.add_argument(
        '--orcagym-addr', default='localhost:50051', help='OrcaGym gRPC address.'
    )
    parser.add_argument(
        '--env-name', default='DoubleGripperTowelControl', help='Gym env name prefix.'
    )
    parser.add_argument(
        '--agent-name', default='NoRobot', help='Agent name used by OrcaGym.'
    )
    parser.add_argument(
        '--seconds', type=float, default=5.5, help='Simulation horizon in seconds.'
    )
    parser.add_argument(
        '--frame-skip', type=int, default=20, help='Frame skip passed to OrcaGym env.'
    )
    parser.add_argument(
        '--time-step', type=float, default=0.001, help='Base simulation timestep.'
    )
    parser.add_argument(
        '--contact-interval',
        type=float,
        default=0.1,
        help='Sampling interval (seconds) for gripper-towel contact logging.',
    )
    parser.add_argument(
        '--contact-output-dir',
        default=None,
        help='Optional output directory for contact logs. Default: auto temp folder.',
    )
    parser.add_argument(
        '--controller-min-steps',
        type=int,
        default=4,
        help='Minimum required auto_step keyframes.',
    )
    parser.add_argument(
        '--realtime',
        action='store_true',
        help='Sleep to approximately match real-time playback.',
    )
    args = parser.parse_args()

    if args.contact_interval <= 0:
        raise ValueError('--contact-interval must be > 0')

    try:
        import gymnasium as gym
        from gymnasium.envs.registration import register, registry
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency 'gymnasium'. Install with: python -m pip install gymnasium"
        ) from exc

    env_id = _register_env(
        gym=gym,
        register=register,
        registry=registry,
        env_name=args.env_name,
        orcagym_addr=args.orcagym_addr,
        frame_skip=args.frame_skip,
        time_step=args.time_step,
        agent_name=args.agent_name,
        controller_min_steps=args.controller_min_steps,
    )
    env = gym.make(env_id)

    try:
        env.reset()
        sim_time = 0.0
        steps = 0
        wall_start = time.time()
        realtime_step = args.time_step * args.frame_skip
        next_sample_time = float(args.contact_interval)

        if args.contact_output_dir:
            contact_output_dir = Path(args.contact_output_dir).expanduser().resolve()
            contact_output_dir.mkdir(parents=True, exist_ok=True)
        else:
            contact_output_dir = Path(
                tempfile.mkdtemp(prefix='double_gripper_towel_contacts_')
            ).resolve()
        contact_file = contact_output_dir / 'gripper_towel_contacts.jsonl'
        contact_samples = 0
        contact_rows = 0

        while sim_time < args.seconds:
            loop_start = time.time()
            _, _, terminated, truncated, info = env.step(None)
            env.render()
            sim_time = float(info.get('sim_time', sim_time))
            steps += 1

            if (
                sim_time + 1e-12 >= next_sample_time
                and next_sample_time <= args.seconds + 1e-12
            ):
                rows = env.unwrapped.sample_gripper_towel_contacts(
                    sample_time=next_sample_time
                )
                record = {
                    'sample_time': float(next_sample_time),
                    'sim_time': float(sim_time),
                    'n_contacts': len(rows),
                    'contacts': rows,
                }
                with contact_file.open('a', encoding='utf-8') as fp:
                    fp.write(json.dumps(record, ensure_ascii=False) + '\n')
                contact_samples += 1
                contact_rows += len(rows)
                next_sample_time += args.contact_interval

            if terminated or truncated:
                env.reset()

            if args.realtime:
                elapsed = time.time() - loop_start
                if elapsed < realtime_step:
                    time.sleep(realtime_step - elapsed)

        summary = {
            'orcagym_addr': args.orcagym_addr,
            'agent_name': args.agent_name,
            'seconds_target': float(args.seconds),
            'seconds_simulated': float(sim_time),
            'steps': int(steps),
            'time_step': float(args.time_step),
            'frame_skip': int(args.frame_skip),
            'wall_time_sec': float(time.time() - wall_start),
            'env_id': env_id,
            'contact_interval': float(args.contact_interval),
            'contact_samples': int(contact_samples),
            'contact_rows': int(contact_rows),
            'contact_output_dir': str(contact_output_dir),
            'contact_file': str(contact_file),
        }
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    finally:
        env.close()


if __name__ == '__main__':
    main()
