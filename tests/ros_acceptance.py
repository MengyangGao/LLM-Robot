"""Opt-in real ROS 2 + Ollama acceptance test; see README for invocation."""
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

os.environ['ROS_DOMAIN_ID'] = str(40 + os.getpid() % 80)
os.environ['ROS_LOCALHOST_ONLY'] = '1'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from turtlesim.msg import Pose

ROOT = Path(__file__).resolve().parents[1]


def main():
    with tempfile.TemporaryFile(mode='w+') as log:
        launch = subprocess.Popen(['ros2', 'launch', 'rosgpt', 'demo.launch.py'],
                                  stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        rclpy.init()
        node = Node('rosgpt_acceptance')
        pose = []
        node.create_subscription(Pose, '/turtle1/pose', lambda msg: pose.__setitem__(slice(None), [msg]), 10)
        pub = node.create_publisher(String, '/voice_cmd', 10)

        def wait(seconds):
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=0.05)

        def snapshot():
            p = pose[0]
            return p.x, p.y, p.theta

        def run_cli(*args):
            completed = subprocess.run(['ros2', 'run', 'rosgpt', 'rosgpt', *args],
                                       cwd=ROOT, capture_output=True, text=True, timeout=210)
            if completed.returncode:
                raise AssertionError(f'CLI failed: {completed.stderr}')
            return json.loads(completed.stdout)

        try:
            deadline = time.monotonic() + 20
            while (not pose or not pub.get_subscription_count()) and time.monotonic() < deadline:
                if launch.poll() is not None:
                    raise AssertionError('ROS launch exited early')
                wait(.1)
            assert pose and pub.get_subscription_count(), 'ROS discovery failed'
            start = snapshot()
            plan = run_cli('前进0.4个单位，然后逆时针旋转45度', '--execute')
            assert [a['action'] for a in plan['actions']] == ['move', 'rotate'], plan
            wait(5)
            end = snapshot()
            distance = math.hypot(end[0] - start[0], end[1] - start[1])
            angle = math.degrees(math.atan2(math.sin(end[2] - start[2]), math.cos(end[2] - start[2])))
            assert .34 < distance < .56 and 40 < angle < 55, (distance, angle)
            print(f'Text → ROS → turtlesim: distance={distance:.3f}, angle={angle:.1f}°')

            start = snapshot()
            plan = run_cli('如果图片里箭头指向左边，逆时针旋转45度；否则停止。',
                           '--image', 'examples/left_arrow.png', '--execute')
            assert plan['actions'] == [{'action': 'rotate', 'angle': 45, 'speed': 30, 'clockwise': False}], plan
            wait(4)
            end = snapshot()
            angle = math.degrees(math.atan2(math.sin(end[2] - start[2]), math.cos(end[2] - start[2])))
            assert 40 < angle < 55, angle
            print(f'Image → ROS → turtlesim: angle={angle:.1f}°')

            msg = String()
            msg.data = json.dumps({'actions': [{'action': 'move', 'distance': 3, 'speed': .5, 'forward': True}]})
            pub.publish(msg)
            wait(.8)
            msg.data = json.dumps({'actions': [{'action': 'stop'}]})
            pub.publish(msg)
            wait(.4)
            stopped = snapshot()
            wait(1)
            end = snapshot()
            assert math.hypot(end[0] - stopped[0], end[1] - stopped[1]) < .03, (stopped, end)
            print('Stop preempts a running action: passed')
        except Exception:
            log.seek(0)
            print(log.read(), file=sys.stderr)
            raise
        finally:
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
            os.killpg(launch.pid, signal.SIGINT)
            try:
                launch.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(launch.pid, signal.SIGKILL)
                launch.wait()


if __name__ == '__main__':
    main()
