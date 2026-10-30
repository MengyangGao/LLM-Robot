"""Plan locally, optionally publish the reviewed plan to ROS 2."""
import argparse
import json
import sys
import time
from .planner import MODEL, plan_instruction


def main(args=None):
    parser = argparse.ArgumentParser(description='Plan turtlesim actions with Ollama')
    parser.add_argument('instruction', help='Natural language instruction')
    parser.add_argument('--model', default=MODEL, help='Ollama model (or set OLLAMA_MODEL)')
    parser.add_argument('--image', help='Optional image file for a vision model')
    parser.add_argument('--execute', action='store_true', help='Publish the plan to /voice_cmd')
    options = parser.parse_args(args)
    try:
        plan = plan_instruction(options.instruction, options.image, options.model)
        print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
        if options.execute:
            import rclpy
            from rclpy.node import Node
            from std_msgs.msg import String
            rclpy.init()
            node = Node('rosgpt_planner')
            try:
                pub = node.create_publisher(String, '/voice_cmd', 10)
                deadline = time.monotonic() + 5
                while pub.get_subscription_count() == 0 and time.monotonic() < deadline:
                    rclpy.spin_once(node, timeout_sec=0.1)
                if pub.get_subscription_count() == 0:
                    raise RuntimeError('No controller subscribed to /voice_cmd')
                msg = String()
                msg.data = json.dumps(plan)
                pub.publish(msg)
                rclpy.spin_once(node, timeout_sec=0.3)
            finally:
                node.destroy_node()
                rclpy.shutdown()
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
