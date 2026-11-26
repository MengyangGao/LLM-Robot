import base64
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

MUJOCO_AVAILABLE = importlib.util.find_spec('mujoco') is not None


@unittest.skipUnless(MUJOCO_AVAILABLE, 'MuJoCo is not installed')
class MujocoTests(unittest.TestCase):
    def setUp(self):
        from rosgpt.mujoco_backend import MujocoMobileSim
        self.sim = MujocoMobileSim()

    def test_moves_rotates_and_brakes(self):
        actions = [
            {'action': 'move', 'distance': .5, 'speed': .5, 'forward': True},
            {'action': 'rotate', 'angle': 90, 'speed': 30, 'clockwise': False},
            {'action': 'move', 'distance': .3, 'speed': .5, 'forward': False},
        ]
        completed = self.sim.run({'actions': actions})
        x, y, yaw = self.sim.pose()
        self.assertEqual([x['action'] for x in completed], ['move', 'rotate', 'move'])
        self.assertTrue(.48 < x < .62, x)
        self.assertTrue(-.42 < y < -.28, y)
        self.assertTrue(math.radians(88) < yaw < math.radians(95), yaw)
        self.assertTrue(max(abs(v) for v in self.sim.data.qvel) < .01)

    def test_stop_discards_subsequent_actions(self):
        completed = self.sim.run({'actions': [
            {'action': 'stop'}, {'action': 'move', 'distance': 1, 'speed': .5, 'forward': True},
        ]})
        self.assertEqual(len(completed), 1)
        self.assertLess(abs(self.sim.pose()[0]), .001)

    def test_square_returns_near_start(self):
        completed = self.sim.run({'actions': [
            {'action': 'square', 'side': .25, 'speed': 1, 'clockwise': False}]})
        x, y, yaw = self.sim.pose()
        self.assertEqual(len(completed), 8)
        self.assertLess(math.hypot(x, y), .08)
        self.assertLess(abs(yaw - 2 * math.pi), .1)

    def test_invalid_plan_rejected_before_motion(self):
        with self.assertRaises(ValueError):
            self.sim.run({'actions': [{'action': 'move', 'distance': 1, 'speed': 99, 'forward': True}]})
        self.assertEqual(self.sim.pose(), (0.0, 0.0, 0.0))

    def test_closed_loop_sends_a_fresh_camera_frame_after_each_action(self):
        from rosgpt.mujoco_harness import run_task
        from rosgpt.planner import plan_instruction

        fixtures = iter([
            {'action': 'rotate', 'angle': 90, 'speed': 30, 'clockwise': False},
            {'action': 'move', 'distance': 1.25, 'speed': .5, 'forward': True},
            {'action': 'stop'},
        ])
        sent_images = []

        def reply(request, timeout):
            payload = json.loads(request.data)
            self.assertEqual(payload['format']['properties']['actions']['maxItems'], 1)
            message = payload['messages'][-1]
            sent_images.append(base64.b64decode(message['images'][0]))
            action = next(fixtures)
            content = json.dumps({'actions': [action]})
            return io.BytesIO(json.dumps({'message': {'content': content}}).encode())

        def plan_next(instruction, image):
            return plan_instruction(instruction, image, backend='mujoco', single_step=True)

        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            with patch.dict(os.environ, {'ROSGPT_PROVIDER': 'ollama'}), \
                 patch('rosgpt.planner.request.urlopen', side_effect=reply):
                report = run_task(self.sim, plan_next, output, 'blue')
            self.assertTrue(report['passed'], report)
            self.assertEqual(len(sent_images), 3)
            self.assertEqual(len(set(sent_images)), 3)
            self.assertEqual(sent_images, [(output / item['image']).read_bytes()
                                           for item in report['rounds']])

    def test_saved_plan_cli_and_simulated_camera(self):
        with tempfile.TemporaryDirectory() as temp:
            plan = Path(temp) / 'plan.json'
            image = Path(temp) / 'camera.png'
            plan.write_text(json.dumps({'actions': [
                {'action': 'move', 'distance': .3, 'speed': .5, 'forward': True}]}))
            env = dict(os.environ, MUJOCO_GL='egl')
            run = subprocess.run([sys.executable, '-m', 'rosgpt.mujoco_demo', '--plan', str(plan),
                                  '--execute', '--snapshot', str(image)],
                                 capture_output=True, text=True, env=env, timeout=30)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertIn('"completed"', run.stdout)
            self.assertTrue(image.read_bytes().startswith(b'\x89PNG\r\n\x1a\n'))


if __name__ == '__main__':
    unittest.main()
