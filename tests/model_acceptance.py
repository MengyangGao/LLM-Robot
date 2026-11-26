"""Exercise the selected real model with text and a closed-loop MuJoCo task."""
import argparse
import json
from pathlib import Path
import sys

from rosgpt.mujoco_backend import MujocoMobileSim
from rosgpt.mujoco_harness import run_task
from rosgpt.planner import MODEL, plan_instruction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=MODEL)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--swap', action='store_true', help='Put the blue target to the right')
    args = parser.parse_args()
    args.out.mkdir(parents=True)

    instruction = 'Move forward 0.5 units at speed 0.5. Return a single move action.'
    text = {'instruction': instruction}
    try:
        plan = plan_instruction(instruction, model=args.model, backend='mujoco')
        text['plan'] = plan
        if [action['action'] for action in plan['actions']] != ['move']:
            raise ValueError('Expected exactly one move')
        sim = MujocoMobileSim()
        pose = sim.run(plan)[0]['pose']
        text['final_pose'] = pose
        text['passed'] = abs(pose[0] - .5) <= .03 and abs(pose[1]) <= .03
    except (ValueError, RuntimeError, OSError) as exc:
        text.update(passed=False, error=str(exc))
    (args.out / 'text.json').write_text(json.dumps(text, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(text, indent=2), flush=True)

    vision_dir = args.out / 'vision'
    vision_dir.mkdir()

    def plan_next(prompt, image):
        return plan_instruction(prompt, image_path=image, model=args.model, backend='mujoco', single_step=True)

    vision = run_task(MujocoMobileSim(), plan_next, vision_dir, 'blue', args.swap)
    vision['model'] = args.model
    summary = {'model': args.model, 'text_passed': text['passed'], 'vision_passed': vision['passed']}
    (vision_dir / 'report.json').write_text(json.dumps(vision, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))
    return 0 if text['passed'] and vision['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
