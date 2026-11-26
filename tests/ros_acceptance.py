"""Exercise the installed ROS 2 demo, optionally including Ollama inference."""
import argparse
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline', action='store_true', help='Use saved plans without Ollama')
    args = parser.parse_args()
    with tempfile.TemporaryFile(mode='w+') as log:
        rclpy.init()
        node = Node('rosgpt_acceptance')
        pose = []
        samples = []

        def on_pose(msg):
            pose[:] = [msg]
            samples.append((msg.x, msg.y, msg.theta))

        node.create_subscription(Pose, '/turtle1/pose', on_pose, 10)
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
                raise AssertionError(f'CLI failed: {completed.stderr}{completed.stdout}')
            # The CLI prints the plan before initializing ROS. DDS may then
            # append its native diagnostic messages to stdout.
            output = completed.stdout.lstrip()
            plan, end = json.JSONDecoder().raw_decode(output)
            if output[end:].strip():
                print(output[end:].strip(), file=sys.stderr)
            return plan

        launch = None
        try:
            launch = subprocess.Popen(
                ['ros2', 'launch', 'rosgpt', 'demo.launch.py'],
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
            deadline = time.monotonic() + 20
            while (not pose or not pub.get_subscription_count()) and time.monotonic() < deadline:
                if launch.poll() is not None:
                    raise AssertionError('ROS launch exited early')
                wait(.1)
            assert pose and pub.get_subscription_count(), 'ROS discovery failed'
            start = snapshot()
            if args.offline:
                plan = run_cli('--execute-plan', 'examples/move_and_turn.json')
            else:
                plan = run_cli('前进0.4个单位，然后逆时针旋转45度', '--execute')
            assert [a['action'] for a in plan['actions']] == ['move', 'rotate'], plan
            wait(5)
            end = snapshot()
            distance = math.hypot(end[0] - start[0], end[1] - start[1])
            angle = math.degrees(math.atan2(math.sin(end[2] - start[2]), math.cos(end[2] - start[2])))
            expected_distance = .5 if args.offline else .4
            assert abs(distance - expected_distance) < .12 and 40 < angle < 55, (distance, angle)
            print(f'ROS → turtlesim: distance={distance:.3f}, angle={angle:.1f}°')

            if not args.offline:
                start = snapshot()
                plan = run_cli('如果图片里箭头指向左边，逆时针旋转45度；否则停止。',
                               '--image', 'examples/left_arrow.png', '--execute')
                assert len(plan['actions']) == 1, plan
                action = plan['actions'][0]
                assert action['action'] == 'rotate' and action['angle'] == 45 and not action['clockwise'], plan
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

            start = snapshot()
            msg.data = json.dumps({'actions': [{'action': 'move', 'distance': .2, 'speed': 1, 'forward': True}]})
            pub.publish(msg)
            wait(1.5)
            end = snapshot()
            assert .15 < math.hypot(end[0] - start[0], end[1] - start[1]) < .35, (start, end)
            print('Integer speed is accepted by the ROS message: passed')

            start = snapshot()
            if args.offline:
                plan = run_cli('--execute-plan', 'examples/red_square.json')
            else:
                plan = run_cli('先把画笔设为红色，然后画一个边长0.4的正方形，速度每秒1个单位，逆时针。', '--execute')
            assert [a['action'] for a in plan['actions']] == ['set_pen', 'square'], plan
            assert (plan['actions'][0]['r'], plan['actions'][0]['g'], plan['actions'][0]['b']) == (255, 0, 0), plan
            offset = len(samples)
            wait(20)
            trace = samples[offset:]
            end = snapshot()
            assert trace, 'no pose feedback during square'
            span_x = max(p[0] for p in trace) - min(p[0] for p in trace)
            span_y = max(p[1] for p in trace) - min(p[1] for p in trace)
            assert .3 < span_x < .7 and .3 < span_y < .7, (span_x, span_y)
            assert math.hypot(end[0] - start[0], end[1] - start[1]) < .2, (start, end)
            print(f'Red pen service + square: trajectory spans {span_x:.2f} × {span_y:.2f}')

            with tempfile.TemporaryDirectory(prefix='llm-robot-plan-') as temp:
                plan_path = str(Path(temp) / 'reviewed.json')
                if args.offline:
                    saved = {'actions': [{'action': 'stop'}]}
                    Path(plan_path).write_text(json.dumps(saved), encoding='utf-8')
                else:
                    saved = run_cli('停止', '--save-plan', plan_path)
                assert saved['actions'] == [{'action': 'stop'}], saved
                executed = run_cli('--execute-plan', plan_path)
                assert executed == saved
                print('Saved plan reviewed and executed without regeneration: passed')
        except Exception:
            log.seek(0)
            print(log.read(), file=sys.stderr)
            raise
        finally:
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
            if launch is not None:
                os.killpg(launch.pid, signal.SIGINT)
                try:
                    launch.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(launch.pid, signal.SIGKILL)
                    launch.wait()


if __name__ == '__main__':
    main()
