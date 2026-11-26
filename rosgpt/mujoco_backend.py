"""Follow planar position references with the Menagerie TidyBot base."""
import math
import os
from contextlib import nullcontext
from importlib.resources import files
from pathlib import Path
import time

# EGL renders offscreen without capturing the desktop when no display is available.
if not os.environ.get('DISPLAY'):
    os.environ.setdefault('MUJOCO_GL', 'egl')

try:
    import mujoco
except ImportError as exc:
    raise ImportError('MuJoCo is not installed in this Python environment; install the mujoco package') from exc

from .planner import expand_actions


class MujocoMobileSim:
    """Planar TidyBot simulation with the upstream position actuators."""

    def __init__(self):
        xml_path = files('rosgpt').joinpath('assets/mobile_arena.xml')
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

    def pose(self):
        return tuple(float(v) for v in self.data.qpos[:3])

    def snapshot(self, path):
        """Render only the simulated camera, never the desktop."""
        from PIL import Image
        with mujoco.Renderer(self.model, height=480, width=640) as renderer:
            renderer.update_scene(self.data, camera='overhead')
            Image.fromarray(renderer.render()).save(Path(path))

    def _step(self, controls, viewer=None):
        if viewer:
            if not viewer.is_running():
                raise RuntimeError('MuJoCo viewer was closed')
        # The passive viewer runs on another thread and shares model/data.
        with viewer.lock() if viewer else nullcontext():
            self.data.ctrl[:] = controls
            mujoco.mj_step(self.model, self.data)
        if viewer:
            viewer.sync()
            time.sleep(self.model.opt.timestep)

    def run(self, plan, viewer=None):
        actions = expand_actions(plan, backend='mujoco')
        completed = []
        timestep = self.model.opt.timestep
        try:
            for action in actions:
                kind = action['action']
                if kind == 'stop':
                    completed.append({'action': kind, 'pose': self.pose()})
                    break
                x, y, yaw = self.pose()
                if kind == 'move':
                    distance = action['distance']
                    duration = distance / action['speed']
                    signed = distance if action['forward'] else -distance
                    delta = (signed * math.cos(yaw), signed * math.sin(yaw), 0)
                else:
                    angle = action['angle']
                    duration = angle / action['speed']
                    signed = angle if not action['clockwise'] else -angle
                    delta = (0, 0, math.radians(signed))
                origin = (x, y, yaw)
                target = tuple(start + change for start, change in zip(origin, delta))
                steps = max(1, math.ceil(duration / timestep))
                for step in range(1, steps + 1):
                    fraction = step / steps
                    self._step(tuple(start + fraction * change
                                     for start, change in zip(origin, delta)), viewer)
                for _ in range(round(.5 / timestep)):
                    self._step(target, viewer)
                if max(abs(actual - expected) for actual, expected in zip(self.pose(), target)) > .02:
                    raise RuntimeError(f'{kind} did not reach its target pose')
                completed.append({'action': kind, 'pose': self.pose()})
        finally:
            with viewer.lock() if viewer else nullcontext():
                self.data.ctrl[:] = self.pose()
        return completed
