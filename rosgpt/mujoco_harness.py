"""Run and score a closed-loop TidyBot color waypoint task."""
import argparse
import json
import math
from pathlib import Path
import sys

from .planner import MODEL, plan_instruction, validate_plan

MAX_ROUNDS = 6
GOAL_TOLERANCE = .08


def waypoint_action(pose, waypoint):
    dx = waypoint.pos[0] - pose[0]
    dy = waypoint.pos[1] - pose[1]
    distance = math.hypot(dx, dy)
    if distance <= GOAL_TOLERANCE:
        return {'action': 'stop'}
    turn = math.atan2(dy, dx) - pose[2]
    angle = math.degrees(math.atan2(math.sin(turn), math.cos(turn)))
    if abs(angle) > 2:
        return {'action': 'rotate', 'angle': round(abs(angle), 2), 'speed': 30, 'clockwise': angle < 0}
    return {'action': 'move', 'distance': round(min(distance, 5), 2), 'speed': .5, 'forward': True}


def run_task(sim, plan_next, output, color, swap=False):
    east = sim.model.geom('red_marker')
    north = sim.model.geom('blue_marker')
    if swap:
        east_color = east.rgba.copy()
        east.rgba[:] = north.rgba
        north.rgba[:] = east_color
    target = east if (color == 'red') != swap else north
    goal = tuple(float(value) for value in target.pos[:2])
    history = []
    report = {'color': color, 'swap': swap, 'goal_xy': goal, 'rounds': history}
    try:
        for index in range(MAX_ROUNDS):
            pose = sim.pose()
            image = output / f'round-{index + 1}.png'
            sim.snapshot(image)
            east_action = waypoint_action(pose, east)
            north_action = waypoint_action(pose, north)
            instruction = (
                f'Find the {color} disk in this image. Is it on the right or at the top? '
                'Use only the image to identify its location. Copy the corresponding action exactly. '
                f'If the {color} disk is on the right, return {json.dumps({"actions": [east_action]})}. '
                f'If the {color} disk is at the top, return {json.dumps({"actions": [north_action]})}.'
            )
            record = {'image': image.name, 'pose_before': pose, 'instruction': instruction}
            history.append(record)
            plan = validate_plan(plan_next(instruction, image), backend='mujoco')
            actions = plan['actions']
            if len(actions) != 1 or actions[0]['action'] not in ('move', 'rotate', 'stop'):
                raise ValueError('Each round requires one move, rotate or stop action')
            if actions[0] not in (east_action, north_action):
                raise ValueError('The returned action did not match either waypoint candidate')
            record['action'] = actions[0]
            record['pose_after'] = sim.run(plan)[0]['pose']
            if actions[0]['action'] == 'stop':
                break
    except (ValueError, RuntimeError, OSError) as exc:
        report['error'] = str(exc)

    report['final_distance'] = math.dist(sim.pose()[:2], goal)
    report['stopped'] = bool(history and history[-1].get('pose_after') is not None
                             and history[-1]['action']['action'] == 'stop')
    report['passed'] = report['stopped'] and report['final_distance'] <= GOAL_TOLERANCE
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--color', choices=('red', 'blue'), default='blue')
    parser.add_argument('--swap', action='store_true', help='Swap the two waypoint colors')
    parser.add_argument('--model', default=MODEL)
    parser.add_argument('--replay', type=Path, help='Read replay actions from a JSON action plan')
    parser.add_argument('--out', required=True, type=Path, help='New directory for images and report')
    options = parser.parse_args()

    try:
        from .mujoco_backend import MujocoMobileSim

        if options.replay:
            plan = validate_plan(json.loads(options.replay.read_text(encoding='utf-8')), backend='mujoco')
            replay = iter(plan['actions'])

            def plan_next(instruction, image):
                action = next(replay, None)
                if action is None:
                    raise ValueError('Replay ended before stop')
                return {'actions': [action]}
        else:
            def plan_next(instruction, image):
                return plan_instruction(instruction, image, options.model, backend='mujoco', single_step=True)

        sim = MujocoMobileSim()
        options.out.mkdir(parents=True)
        report = run_task(sim, plan_next, options.out, options.color, options.swap)
        report['planner'] = 'replay' if options.replay else options.model
        result = json.dumps(report, indent=2)
        (options.out / 'report.json').write_text(result + '\n', encoding='utf-8')
        print(result)
        return 0 if report['passed'] else 1
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
