"""Run a local language or vision plan in the MuJoCo mobile-base simulation."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import tempfile
import threading

from .planner import MODEL, plan_instruction, validate_plan


def main(args=None):
    parser = argparse.ArgumentParser(description='Ollama + MuJoCo mobile robot simulation')
    parser.add_argument('instruction', nargs='?', help='Natural language instruction')
    parser.add_argument('--plan', help='Load an existing JSON plan instead of calling the model')
    parser.add_argument('--model', default=MODEL, help='Model ID for the selected planner provider')
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument('--image', help='Image file to send to the vision model')
    inputs.add_argument('--observe', action='store_true', help='Use the MuJoCo overhead camera')
    parser.add_argument('--execute', action='store_true', help='Execute the validated plan in simulation')
    parser.add_argument('--view', action='store_true', help='Open the MuJoCo interactive viewer while executing')
    parser.add_argument('--snapshot', help='Save the initial simulated camera image as PNG')
    options = parser.parse_args(args)
    if bool(options.instruction) == bool(options.plan):
        parser.error('provide an instruction or --plan')
    if options.plan and (options.image or options.observe):
        parser.error('--plan cannot be combined with image input')
    if options.view and not options.execute:
        parser.error('--view requires --execute')
    try:
        from .mujoco_backend import MujocoMobileSim
        sim = MujocoMobileSim()
        with tempfile.TemporaryDirectory(prefix='llm-robot-') as temp:
            camera_path = Path(temp) / 'camera.png'
            if options.observe or options.snapshot:
                sim.snapshot(options.snapshot or camera_path)
            image_path = options.image
            if options.observe:
                image_path = str(options.snapshot or camera_path)
            plan = (validate_plan(json.loads(Path(options.plan).read_text(encoding='utf-8')), backend='mujoco')
                    if options.plan else plan_instruction(options.instruction, image_path, options.model, backend='mujoco'))
            print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
            if options.execute:
                viewer_threads = []
                if options.view:
                    import mujoco.viewer
                    threads_before = set(threading.enumerate())
                    context = mujoco.viewer.launch_passive(sim.model, sim.data)
                    # MuJoCo's passive viewer owns a daemon render thread. Wait
                    # for that thread after close so GLFW is not torn down while
                    # it is still drawing.
                    viewer_threads = [thread for thread in threading.enumerate()
                                      if thread not in threads_before and thread.daemon
                                      and '_launch_internal' in thread.name]
                else:
                    context = nullcontext(None)
                try:
                    with context as viewer:
                        completed = sim.run(plan, viewer)
                finally:
                    for thread in viewer_threads:
                        thread.join(timeout=5)
                print(json.dumps({'completed': completed}, ensure_ascii=False, indent=2), flush=True)
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
