"""Plan locally, optionally publish the reviewed plan to ROS 2."""
import argparse
import json
from pathlib import Path
import sys
import time
from .planner import MODEL, plan_instruction, validate_plan


def publish_plan(plan):
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


def main(args=None):
    parser = argparse.ArgumentParser(description='Plan turtlesim actions with Ollama')
    parser.add_argument('instruction', nargs='?', help='Natural language instruction')
    parser.add_argument('--model', default=MODEL, help='Model ID for the selected planner provider')
    parser.add_argument('--image', help='Optional image file for a vision model')
    parser.add_argument('--execute', action='store_true', help='Publish the plan to /voice_cmd')
    parser.add_argument('--save-plan', help='Save the generated plan as JSON for review')
    parser.add_argument('--execute-plan', help='Validate and publish a previously saved JSON plan')
    options = parser.parse_args(args)
    if bool(options.instruction) == bool(options.execute_plan):
        parser.error('provide an instruction or --execute-plan')
    if options.execute_plan and (options.image or options.save_plan or options.execute):
        parser.error('--execute-plan cannot be combined with --image, --save-plan or --execute')
    try:
        if options.execute_plan:
            plan = validate_plan(json.loads(Path(options.execute_plan).read_text(encoding='utf-8')))
        else:
            plan = plan_instruction(options.instruction, options.image, options.model)
        print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
        if options.save_plan:
            Path(options.save_plan).write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        if options.execute or options.execute_plan:
            publish_plan(plan)
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
