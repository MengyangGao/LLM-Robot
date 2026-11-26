"""Use the pinned official LeRobot SO-101 tabletop environment in simulation."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile

from .planner import MODEL, plan_instruction, validate_plan

REPOSITORY = 'lerobot/so101-tabletop-mujoco'
REVISION = '844dee6c058960c2ef42b65e2c68b47a5cf7c4f3'
DEFAULT_DIR = Path.home() / '.cache' / 'llm-robot' / 'so101-844dee6c'


def download_environment(directory):
    from huggingface_hub import snapshot_download
    snapshot_download(REPOSITORY, revision=REVISION, local_dir=str(directory), allow_patterns=[
        'env.py', 'so101_tabletop.xml', 'robotstudio_so101/*', 'robotstudio_so101/**/*',
        'textures/*', 'README.md',
    ])


def load_environment(directory):
    """Load a local copy of the official environment, which contains Python code."""
    source = Path(directory) / 'env.py'
    if not source.is_file():
        raise FileNotFoundError(f'SO-101 environment not found at {source}; use --download')
    spec = importlib.util.spec_from_file_location('llm_robot_so101_upstream', source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main(args=None):
    parser = argparse.ArgumentParser(description='LeRobot SO-101 tabletop simulation')
    parser.add_argument('instruction', nargs='?', help='Natural language instruction')
    parser.add_argument('--plan', type=Path, help='Load a reviewed SO-101 JSON plan')
    parser.add_argument('--model', default=MODEL, help='Model ID for the selected planner provider')
    parser.add_argument('--env-dir', type=Path, default=DEFAULT_DIR, help='Trusted local environment directory')
    parser.add_argument('--download', action='store_true', help='Download the pinned official SO-101 environment')
    parser.add_argument('--observe', action='store_true', help='Send the simulated front camera to the model')
    parser.add_argument('--snapshot', type=Path, help='Save the initial simulated camera image')
    parser.add_argument('--final-snapshot', type=Path, help='Save the camera image after execution')
    parser.add_argument('--execute', action='store_true', help='Execute the validated plan in simulation')
    options = parser.parse_args(args)
    if bool(options.instruction) == bool(options.plan):
        parser.error('provide an instruction or --plan')
    if options.plan and options.observe:
        parser.error('--plan cannot be combined with --observe')
    if options.final_snapshot and not options.execute:
        parser.error('--final-snapshot requires --execute')
    if not os.environ.get('DISPLAY'):
        os.environ.setdefault('MUJOCO_GL', 'egl')
    env = None
    try:
        if options.download:
            download_environment(options.env_dir)
        upstream = load_environment(options.env_dir)
        import numpy as np
        from PIL import Image
        env = upstream.SO101Env(observation_height=240, observation_width=320, cameras={'front': 'front'})
        observation, _ = env.reset(seed=0)
        with tempfile.TemporaryDirectory(prefix='llm-robot-so101-') as temp:
            camera = options.snapshot or Path(temp) / 'front.png'
            if options.observe or options.snapshot:
                Image.fromarray(observation['pixels']['front']).save(camera)
            plan = (validate_plan(json.loads(options.plan.read_text(encoding='utf-8')), backend='so101')
                    if options.plan else plan_instruction(options.instruction, str(camera) if options.observe else None,
                                                          options.model, backend='so101'))
            print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
            if options.execute:
                target = observation['agent_pos'].copy()
                completed = []
                for action in plan['actions']:
                    if action['action'] == 'stop':
                        completed.append({'action': 'stop'})
                        break
                    if action['action'] == 'home':
                        target = np.asarray(upstream.HOME, dtype=np.float32)
                    else:
                        target[5] = float(action['position'])
                    for _ in range(45):
                        observation, reward, terminated, truncated, info = env.step(target)
                        if terminated or truncated:
                            raise RuntimeError('SO-101 simulation episode ended before the plan completed')
                    completed.append({'action': action['action'],
                                      'joint_positions': [float(x) for x in observation['agent_pos']]})
                if options.final_snapshot:
                    Image.fromarray(observation['pixels']['front']).save(options.final_snapshot)
                print(json.dumps({'completed': completed}, ensure_ascii=False, indent=2), flush=True)
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 1
    finally:
        if env is not None:
            env.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
