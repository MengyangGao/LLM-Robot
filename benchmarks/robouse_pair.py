"""Compare Show-Harness subgoal planning against direct action selection."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

TASKS = {
    'development': 'arc-gravity-vision',
    'holdout': 'arc-stack-by-size-vision',
}
CONDITIONS = {
    'baseline': {'mode': 'zs'},
    'direct': {'mode': 'zs', 'plugins': {'subgoal': False}},
}


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')


def read_trial(directory):
    results = list(directory.glob('*/result.json'))
    if len(results) != 1:
        raise RuntimeError(f'Expected one trial result in {directory}, found {len(results)}')
    result = json.loads(results[0].read_text(encoding='utf-8'))
    if result.get('status') != 'completed' or result.get('exception_info'):
        raise RuntimeError(f'Upstream trial did not complete cleanly: {results[0]}')
    if result.get('image_input') is not True:
        raise RuntimeError(f'Upstream trial did not enable image input: {results[0]}')
    episode = result.get('episode') or {}
    verifier = result.get('verifier_result') or {}
    reward = verifier.get('rewards', {}).get('reward')
    if type(reward) not in (float, int):
        raise RuntimeError(f'Upstream verifier did not return a numeric reward: {results[0]}')
    decisions = 0
    empty_grasps = 0
    motions_under_1mm = 0
    log = results[0].parent / 'agent' / 'stdout.jsonl'
    if log.is_file():
        with log.open(encoding='utf-8') as stream:
            for line in stream:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict) and event.get('type') == 'decision':
                    decisions += 1
                    action_result = event.get('result') or {}
                    empty_grasps += bool(action_result.get('grasp_empty'))
                    displacement = action_result.get('moved_m')
                    if action_result.get('kind') == 'move' and isinstance(displacement, (int, float)):
                        motions_under_1mm += displacement < .001
    return {
        'reward': reward,
        'success': episode.get('success'),
        'progress': (episode.get('metrics') or {}).get('progress'),
        'steps': episode.get('steps_used'),
        'decisions': decisions,
        'empty_grasps': empty_grasps,
        'motions_under_1mm': motions_under_1mm,
        'outcome': episode.get('outcome'),
        'image_input': result.get('image_input'),
        'wall_time_s': result.get('wall_time_s'),
        'result': str(results[0]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robouse', default='robouse')
    parser.add_argument('--model', default='qwen3-vl:2b-instruct')
    parser.add_argument('--ollama-url', default='http://127.0.0.1:11434/v1')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--timeout', type=int, default=90, help='Model wall-clock budget per trial')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    out = args.out.resolve()
    out.mkdir(parents=True)
    protocol = {
        'task_splits': TASKS, 'conditions': CONDITIONS, 'seed': args.seed, 'model': args.model,
        'harness': 'show-harness', 'timeout_s': args.timeout, 'primary_metric': 'upstream reward',
    }
    write_json(out / 'protocol.json', protocol)
    for name, config in CONDITIONS.items():
        write_json(out / f'{name}.json', config)

    results = {}
    env = dict(os.environ, ROBOUSE_SHOW_HARNESS_URL=args.ollama_url)
    env.setdefault('MUJOCO_GL', 'egl')
    for split, task in TASKS.items():
        pair = {}
        results[split] = pair
        # Reverse trial order on holdout to reduce a systematic warm-cache advantage.
        order = CONDITIONS if split == 'development' else reversed(CONDITIONS)
        for condition in order:
            destination = out / split / condition
            destination.mkdir(parents=True)
            command = [
                args.robouse, 'run', '--task', task, '--harness', 'show-harness',
                '--model', args.model, '--seed', str(args.seed), '--timeout', str(args.timeout),
                '--out', str(destination),
            ]
            print(shlex.join(command), flush=True)
            trial_env = dict(env, ROBOUSE_SHOW_HARNESS_CONFIG=str(out / f'{condition}.json'))
            start = time.monotonic()
            with (destination / 'runner.log').open('w', encoding='utf-8') as log:
                completed = subprocess.run(command, env=trial_env, stdout=log, stderr=subprocess.STDOUT)
            if completed.returncode:
                pair[condition] = {'error': f'Runner exit {completed.returncode}', 'reward': None}
            else:
                try:
                    pair[condition] = read_trial(destination)
                except (OSError, ValueError, RuntimeError) as exc:
                    pair[condition] = {'error': str(exc), 'reward': None}
            pair[condition]['runner_seconds'] = round(time.monotonic() - start, 2)
            write_json(out / 'summary.json', results)
            print(json.dumps(pair[condition], ensure_ascii=False), flush=True)
        baseline, direct = (pair[name]['reward'] for name in CONDITIONS)
        pair['reward_delta'] = direct - baseline if baseline is not None and direct is not None else None
        write_json(out / 'summary.json', results)
    return 1 if any(pair['reward_delta'] is None for pair in results.values()) else 0


if __name__ == '__main__':
    raise SystemExit(main())
