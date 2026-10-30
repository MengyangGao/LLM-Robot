import base64
import json
import tempfile
import unittest
from unittest.mock import patch

from rosgpt.planner import plan_instruction, validate_plan

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
