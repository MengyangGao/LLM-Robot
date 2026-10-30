"""A bounded, timer-driven turtlesim action executor with pose feedback."""
import json
import math
import time
from collections import deque

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from turtlesim.msg import Pose
from .planner import validate_plan


class TurtlesimController(Node):
    def __init__(self):
        super().__init__('turtlesim_controller')
        self.velocity_publisher = self.create_publisher(Twist, '/turtle1/cmd_vel', 10)
        self.create_subscription(String, '/voice_cmd', self.voice_cmd_callback, 10)
        self.create_subscription(Pose, '/turtle1/pose', self.pose_callback, 10)
        self.create_timer(0.05, self.tick)
        self.pose = None
        self.pose_time = 0.0
        self.pending = deque()
        self.active = None
        self.get_logger().info('Waiting for validated plans on /voice_cmd')

    def pose_callback(self, msg):
        self.pose = msg
        self.pose_time = time.monotonic()

    def stop(self):
        self.velocity_publisher.publish(Twist())
        self.active = None
        self.pending.clear()

    def voice_cmd_callback(self, msg):
        try:
            plan = validate_plan(json.loads(msg.data))
        except (ValueError, TypeError) as exc:
            self.get_logger().error(f'Rejected plan: {exc}')
            return
        actions = plan['actions']
        if len(actions) == 1 and actions[0]['action'] == 'stop':
            self.stop()
            return
        if self.active or self.pending:
            self.get_logger().warning('Controller busy; plan rejected')
            return
        if self.pose is None or time.monotonic() - self.pose_time > 1:
            self.get_logger().error('No recent turtlesim pose; plan rejected')
            return
        self.pending.extend(actions)

    def tick(self):
        now = time.monotonic()
        if not self.active and not self.pending:
            return
        if self.pose is None or now - self.pose_time > 1:
            self.get_logger().error('Pose timed out; stopping')
            self.stop()
            return
        if self.active is None:
            action = self.pending.popleft()
            if action['action'] == 'stop':
                self.stop()
                return
            target = action['distance'] if action['action'] == 'move' else action['angle']
            self.active = {'action': action, 'start': (self.pose.x, self.pose.y),
                           'previous_theta': self.pose.theta, 'turned': 0.0,
                           'deadline': now + 2 * target / action['speed'] + 3}
        state = self.active
        action = state['action']
        if now > state['deadline']:
            self.get_logger().error('Action timed out; stopping')
            self.stop()
            return
        twist = Twist()
        if action['action'] == 'move':
            progress = math.hypot(self.pose.x - state['start'][0], self.pose.y - state['start'][1])
            finished = progress >= action['distance']
            twist.linear.x = action['speed'] * (1 if action['forward'] else -1)
        else:
            delta = math.atan2(math.sin(self.pose.theta - state['previous_theta']),
                               math.cos(self.pose.theta - state['previous_theta']))
            state['previous_theta'] = self.pose.theta
            state['turned'] += delta * (-1 if action['clockwise'] else 1)
            finished = state['turned'] >= math.radians(action['angle'])
            twist.angular.z = math.radians(action['speed']) * (-1 if action['clockwise'] else 1)
        if finished:
            self.velocity_publisher.publish(Twist())
            self.active = None
        else:
            self.velocity_publisher.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = TurtlesimController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
