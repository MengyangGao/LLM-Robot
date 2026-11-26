"""Run a calibrated RGB pick-and-place policy on Robo Use's public stacking task."""
import argparse
import base64
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image
from robouse.core.session import Episode
from robouse.tasks import load_task, resolve_task_path

TASK = 'arc-stack-by-size-vision'
COLORS = ('purple', 'orange', 'cyan')
OPEN = -.3
CARRY = .16


def measure(path, projection):
    pixels = np.asarray(Image.open(path).convert('RGB'), dtype=int)
    r, g, b = np.moveaxis(pixels, 2, 0)
    masks = {
        'purple': (r > 225) & (b > 225) & (g < 205),
        'orange': (r > 235) & (g > 155) & (b < 125),
        'cyan': (r < 165) & (g > 225) & (b > 225),
    }

    def table_point(u, v, z):
        ray = projection[:2] - np.array([[u], [v]]) * projection[2]
        return np.linalg.solve(ray[:, :2], -ray[:, 3] - ray[:, 2] * z)

    cubes = {}
    for color, mask in masks.items():
        rows, cols = np.where(mask)
        if len(rows) < 20 or np.ptp(cols) > 32 or np.ptp(rows) > 32:
            raise ValueError(f'No unambiguous {color} block in the top view')
        # The calibrated top camera is nearly vertical. Refine size and
        # center using the top surface height instead of the table plane.
        u, v = (cols.min() + cols.max()) / 2, (rows.min() + rows.max()) / 2
        half_pixels = (np.ptp(cols) + 1) / 2
        edge_z = 0.0
        for _ in range(3):
            edge = np.linalg.norm(table_point(u + half_pixels, v, edge_z)
                                  - table_point(u - half_pixels, v, edge_z))
            edge_z = edge
        cubes[color] = {'xy': table_point(u, v, edge_z).tolist(), 'edge_m': float(edge)}
    return cubes


def infer_order(model, image, examples, cubes):
    sizes = {color: round(cube['edge_m'] * 100, 2) for color, cube in cubes.items()}
    question = (
        'Infer the rule from the before and after examples. For the current three blocks, '
        'decide whether the example stacks increase or decrease in size from bottom to top. Level 0 means bottom. '
        f'Examples: {json.dumps(examples)}. Current measured cube edges in cm: {json.dumps(sizes)}. '
        'Answer JSON with order equal to descending (largest at bottom) or ascending (smallest at bottom).'
    )
    schema = {
        'type': 'object', 'required': ['order'], 'additionalProperties': False,
        'properties': {'order': {'enum': ['descending', 'ascending']}},
    }
    payload = {
        'model': model, 'stream': False, 'think': False, 'format': schema,
        'options': {'temperature': 0, 'num_predict': 128},
        'messages': [{'role': 'user', 'content': question,
                      'images': [base64.b64encode(image.read_bytes()).decode('ascii')]}],
    }
    base = os.environ.get('OLLAMA_HOST', 'http://127.0.0.1:11434')
    request = Request(base.rstrip('/') + '/api/chat', data=json.dumps(payload).encode(),
                      headers={'Content-Type': 'application/json'})
    with urlopen(request, timeout=180) as response:
        order = json.loads(json.load(response)['message']['content'])['order']
    if order not in ('descending', 'ascending'):
        raise ValueError('Model must return descending or ascending')
    return order


def run_trial(output, seed, model):
    task = load_task(resolve_task_path(TASK))
    episode = Episode(task.spec, output, seed=seed)

    def call(op, **kwargs):
        response = episode.handle({'op': op, **kwargs})
        if not response['ok']:
            raise RuntimeError(response['error'])
        return response['result']

    def top():
        return Path(call('observe', image=True, camera='top')['image_path'])

    def move(x, y, z, grip, tol=.003):
        result = call('move_to', pos=[x, y, z], grip=grip, max_steps=140, tol=tol)
        if not result['reached']:
            raise RuntimeError(f"Move failed with {result['distance']:.3f} m remaining")

    try:
        projection = np.array(next(cam['projection'] for cam in call('info')['cameras']
                                   if cam['name'] == 'top'))
        observation = call('observe', image=True)
        image = Path(observation['image_path'][-1])
        cubes = measure(image, projection)
        rule = infer_order(model, image, observation['state']['examples'], cubes)
        order = sorted(cubes, key=lambda color: cubes[color]['edge_m'], reverse=rule == 'descending')
        (output / 'plan.json').write_text(json.dumps({
            'model': model, 'image': image.name, 'cubes': cubes, 'rule': rule, 'order_bottom_to_top': order,
        }, indent=2) + '\n', encoding='utf-8')
        x_goal, y_goal = cubes[order[0]]['xy']
        surface = cubes[order[0]]['edge_m']
        for color in order[1:]:
            x, y = cubes[color]['xy']
            half = cubes[color]['edge_m'] / 2
            move(x, y, CARRY, OPEN, .005)
            move(x, y, max(half, .012), OPEN)
            state = call('grip', value=1.0, steps=12)['state']
            if not state.get('holding'):
                raise RuntimeError(f'Empty grasp of {color}')
            move(x, y, CARRY, 1.0, .005)
            top()
            move(x_goal, y_goal, CARRY, 1.0)
            move(x_goal, y_goal, surface + half + .004, 1.0)
            call('grip', value=OPEN, steps=12)
            move(x_goal, y_goal, CARRY, OPEN, .005)
            top()
            surface += 2 * half
        call('done', text='Executed the inferred stack order and released both blocks')
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        if not episode.finished:
            call('give_up', text=str(exc))
    finally:
        episode.close()
    return episode.result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--model', default='qwen3-vl:2b-instruct')
    parser.add_argument('--seed', type=int, action='append')
    args = parser.parse_args()
    os.environ.setdefault('MUJOCO_GL', 'egl')
    args.out.mkdir(parents=True)
    results = [run_trial(args.out / f'seed-{seed}', seed, args.model)
               for seed in dict.fromkeys(args.seed or [0])]
    (args.out / 'summary.json').write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
    for result in results:
        print(json.dumps({key: result[key] for key in ('seed', 'success', 'steps_used', 'agent_text')}))
    return 0 if all(result['success'] for result in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
