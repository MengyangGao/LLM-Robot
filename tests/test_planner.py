import base64
import json
import os
import tempfile
import unittest
from unittest.mock import patch

from rosgpt.planner import expand_actions, plan_instruction, validate_plan

MOVE = {'action': 'move', 'distance': 1, 'speed': .5, 'forward': True}
ROTATE = {'action': 'rotate', 'angle': 90, 'speed': 30, 'clockwise': False}


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self):
        return json.dumps({'message': {'content': json.dumps({'actions': [MOVE, ROTATE]})}}).encode()


class PlannerTests(unittest.TestCase):
    def test_valid_sequence(self):
        self.assertEqual(validate_plan({'actions': [MOVE, ROTATE, {'action': 'stop'}]})['actions'][0], MOVE)

    def test_bad_plans_rejected(self):
        bad = [{}, {'actions': []}, {'actions': [{'action': 'dance'}]},
               {'actions': [dict(MOVE, speed=0)]}, {'actions': [dict(MOVE, speed=True)]},
               {'actions': [dict(MOVE, distance=float('nan'))]},
               {'actions': [dict(MOVE, forward='true')]}, {'actions': [dict(MOVE, extra=1)]},
               {'actions': [dict(MOVE, distance=10 ** 400)]},
               {'actions': [dict(ROTATE, angle=361)]}, {'actions': [dict(MOVE, speed=.001)]},
               {'actions': [MOVE] * 11}]
        for plan in bad:
            with self.subTest(plan=plan), self.assertRaises(ValueError):
                validate_plan(plan)

    def test_square_expansion_and_backend_capabilities(self):
        square = {'action': 'square', 'side': .5, 'speed': .5, 'clockwise': False}
        expanded = expand_actions({'actions': [square]}, backend='mujoco')
        self.assertEqual([a['action'] for a in expanded], ['move', 'rotate'] * 4)
        pen = {'action': 'set_pen', 'r': 255, 'g': 0, 'b': 0, 'width': 3, 'enabled': True}
        validate_plan({'actions': [pen]})
        with self.assertRaises(ValueError):
            validate_plan({'actions': [pen]}, backend='mujoco')

    def test_so101_only_accepts_supported_gripper_and_home_actions(self):
        validate_plan({'actions': [{'action': 'home'}, {'action': 'gripper', 'position': 80}]}, backend='so101')
        for action in (MOVE, {'action': 'gripper', 'position': True},
                       {'action': 'gripper', 'position': 101}, {'action': 'home', 'extra': 1}):
            with self.subTest(action=action), self.assertRaises(ValueError):
                validate_plan({'actions': [action]}, backend='so101')

    @patch.dict(os.environ, {'ROSGPT_PROVIDER': 'ollama'})
    @patch('rosgpt.planner.request.urlopen', return_value=FakeResponse())
    def test_model_and_vision_payload(self, urlopen):
        with tempfile.NamedTemporaryFile() as image:
            image.write(b'example image')
            image.flush()
            plan = plan_instruction('move', image.name, model='vision:test')
        payload = json.loads(urlopen.call_args.args[0].data)
        self.assertEqual(payload['model'], 'vision:test')
        self.assertFalse(payload['think'])
        self.assertEqual(base64.b64decode(payload['messages'][1]['images'][0]), b'example image')
        self.assertEqual(plan['actions'], [MOVE, ROTATE])

    @patch.dict(os.environ, {'ROSGPT_PROVIDER': 'openai', 'OPENAI_API_KEY': 'test-key'})
    @patch('rosgpt.planner.request.urlopen')
    def test_openai_vision_plan_uses_same_validator(self, urlopen):
        class OpenAIResponse(FakeResponse):
            def read(self):
                return json.dumps({'choices': [{'message': {'content': self.content}}]}).encode()

        response = OpenAIResponse()
        response.content = json.dumps({'actions': [MOVE]})
        urlopen.return_value = response
        with tempfile.NamedTemporaryFile() as image:
            image.write(b'\x89PNG\r\n\x1a\n')
            image.flush()
            plan = plan_instruction('Move', image.name, model='gpt-6.2-astra', backend='mujoco')
        self.assertEqual(plan['actions'], [MOVE])
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, 'https://api.openai.com/v1/chat/completions')
        payload = json.loads(request.data)
        image_url = payload['messages'][1]['content'][1]['image_url']['url']
        self.assertTrue(image_url.startswith('data:image/png;base64,'))
        response.content = json.dumps({'actions': [dict(MOVE, speed=100)]})
        with self.assertRaises(ValueError):
            plan_instruction('Move', model='gpt-6.2-astra', backend='mujoco')

    @patch.dict(os.environ, {'ROSGPT_PROVIDER': 'ollama'})
    @patch('rosgpt.planner.request.urlopen')
    def test_model_output_is_validated(self, urlopen):
        class Invalid(FakeResponse):
            def read(self):
                return b'{"message":{"content":"{\\"actions\\":[{\\"action\\":\\"dance\\"}]}"}}'
        urlopen.return_value = Invalid()
        with self.assertRaises(ValueError):
            plan_instruction('dance')


if __name__ == '__main__':
    unittest.main()
