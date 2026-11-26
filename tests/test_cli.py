import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rosgpt.rosgpt import main

PLAN = {'actions': [{'action': 'move', 'distance': 1, 'speed': .5, 'forward': True}]}


class CliTests(unittest.TestCase):
    def test_reviewed_plan_can_be_saved_and_executed_without_regenerating(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            plan = Path(temp) / 'plan.json'
            with patch('rosgpt.rosgpt.plan_instruction', return_value=PLAN) as model:
                self.assertEqual(main(['前进一米', '--save-plan', str(plan)]), 0)
                self.assertEqual(json.loads(plan.read_text()), PLAN)
                self.assertEqual(model.call_count, 1)
            with patch('rosgpt.rosgpt.publish_plan') as publisher, \
                 patch('rosgpt.rosgpt.plan_instruction') as model:
                self.assertEqual(main(['--execute-plan', str(plan)]), 0)
                publisher.assert_called_once_with(PLAN)
                model.assert_not_called()

    def test_invalid_saved_plan_is_never_published(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stderr(io.StringIO()):
            plan = Path(temp) / 'plan.json'
            plan.write_text('{"actions": [{"action": "dance"}]}')
            with patch('rosgpt.rosgpt.publish_plan') as publisher:
                self.assertEqual(main(['--execute-plan', str(plan)]), 1)
                publisher.assert_not_called()


if __name__ == '__main__':
    unittest.main()
