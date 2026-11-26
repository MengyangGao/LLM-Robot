import importlib
import json
import math
import sys
import types
import unittest
from collections import deque
from unittest.mock import patch


def module(name, **attrs):
    result = types.ModuleType(name)
    result.__dict__.update(attrs)
    return result


class Twist:
    def __init__(self):
        self.linear = types.SimpleNamespace(x=0.0)
        self.angular = types.SimpleNamespace(z=0.0)


class Publisher:
    def __init__(self):
        self.messages = []

    def publish(self, msg):
        self.messages.append(msg)


class Logger:
    def error(self, msg):
        pass

    def warning(self, msg):
        pass


class SetPen:
    class Request:
        pass


stubs = {
    'rclpy': module('rclpy'), 'rclpy.node': module('rclpy.node', Node=object),
    'geometry_msgs': module('geometry_msgs'), 'geometry_msgs.msg': module('geometry_msgs.msg', Twist=Twist),
    'std_msgs': module('std_msgs'), 'std_msgs.msg': module('std_msgs.msg', String=object),
    'turtlesim': module('turtlesim'), 'turtlesim.msg': module('turtlesim.msg', Pose=object),
    'turtlesim.srv': module('turtlesim.srv', SetPen=SetPen),
}
with patch.dict(sys.modules, stubs):
    controller = importlib.import_module('rosgpt.rosgptparser_turtlesim')


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.node = controller.TurtlesimController.__new__(controller.TurtlesimController)
        self.node.velocity_publisher = Publisher()
        self.node.get_logger = lambda: Logger()
        self.node.pose = types.SimpleNamespace(x=0.0, y=0.0, theta=math.pi - .1)
        self.node.pose_time = 100.0
        self.node.pending = deque()
        self.node.active = None
        self.clock = patch.object(controller.time, 'monotonic', return_value=100.0)
        self.clock.start()
        self.addCleanup(self.clock.stop)

    def send(self, actions):
        self.node.voice_cmd_callback(types.SimpleNamespace(data=json.dumps({'actions': actions})))

    def test_sequence_and_stop_preemption(self):
        move = {'action': 'move', 'distance': 1, 'speed': .5, 'forward': True}
        self.send([move, move])
        self.node.tick()
        self.assertEqual(self.node.velocity_publisher.messages[-1].linear.x, .5)
        self.node.pose.x = 1.1
        self.node.tick()
        self.assertEqual(self.node.velocity_publisher.messages[-1].linear.x, 0)
        self.node.tick()
        self.assertEqual(self.node.velocity_publisher.messages[-1].linear.x, .5)
        self.send([{'action': 'stop'}])
        self.assertIsNone(self.node.active)
        self.assertFalse(self.node.pending)
        self.assertEqual(self.node.velocity_publisher.messages[-1].linear.x, 0)

    def test_rotation_crosses_pi_and_stops(self):
        self.send([{'action': 'rotate', 'angle': 20, 'speed': 30, 'clockwise': False}])
        self.node.tick()
        self.node.pose.theta = -math.pi + .3
        self.node.tick()
        self.assertIsNone(self.node.active)
        self.assertEqual(self.node.velocity_publisher.messages[-1].angular.z, 0)

    def test_stale_pose_stops_and_discards_queue(self):
        self.send([{'action': 'move', 'distance': 1, 'speed': .5, 'forward': True}])
        self.node.tick()
        self.node.pose_time = 98
        self.node.tick()
        self.assertIsNone(self.node.active)
        self.assertEqual(self.node.velocity_publisher.messages[-1].linear.x, 0)

    def test_timeout_stops_and_rejects_new_plan_while_busy(self):
        move = {'action': 'move', 'distance': 1, 'speed': .5, 'forward': False}
        self.send([move])
        self.node.tick()
        self.assertEqual(self.node.velocity_publisher.messages[-1].linear.x, -.5)
        self.send([move])
        self.assertFalse(self.node.pending)
        self.node.active['deadline'] = 99
        self.node.tick()
        self.assertIsNone(self.node.active)
        self.assertEqual(self.node.velocity_publisher.messages[-1].linear.x, 0)

    def test_invalid_command_is_not_executed(self):
        self.send([{'action': 'move', 'distance': 1, 'speed': 100, 'forward': True}])
        self.node.tick()
        self.assertFalse(self.node.pending)
        self.assertFalse(self.node.velocity_publisher.messages)

    def test_integer_speed_is_sent_as_ros_float(self):
        self.send([{'action': 'move', 'distance': 1, 'speed': 1, 'forward': True}])
        self.node.tick()
        self.assertIs(type(self.node.velocity_publisher.messages[-1].linear.x), float)

    def test_pen_service_completes_before_square_starts(self):
        future = types.SimpleNamespace(done=lambda: True, result=lambda: None)
        calls = []
        self.node.pen_client = types.SimpleNamespace(service_is_ready=lambda: True,
            call_async=lambda request: calls.append(request) or future)
        self.send([{'action': 'set_pen', 'r': 255, 'g': 0, 'b': 0, 'width': 3, 'enabled': True},
                   {'action': 'square', 'side': .5, 'speed': .5, 'clockwise': False}])
        self.node.tick()
        self.assertEqual((calls[0].r, calls[0].g, calls[0].b, calls[0].off), (255, 0, 0, 0))
        self.node.tick()
        self.assertIsNone(self.node.active)
        self.node.tick()
        self.assertEqual(self.node.velocity_publisher.messages[-1].linear.x, .5)
        self.assertEqual(len(self.node.pending), 7)


if __name__ == '__main__':
    unittest.main()
