import numpy as np
from nose2 import tools
import os

import logging
logger = logging.getLogger(__name__)

from gym import envs
from gym.monitoring.tests import helpers

specs = [spec for spec in envs.registry.all() if spec._entry_point is not None]
@tools.params(*specs)
def test_renderable_after_monitor_close(spec):
    # TODO(gdb 2016-05-15): Re-enable these tests after fixing box2d-py
    if spec._entry_point.startswith('gym.envs.box2d:'):
        logger.warn("Skipping tests for box2d env {}".format(spec._entry_point))
        return
    elif spec._entry_point.startswith('gym.envs.parameter_tuning:'):
        logger.warn("Skipping tests for parameter tuning".format(spec._entry_point))
        return

    # Skip mujoco tests
    skip_mujoco = not (os.environ.get('MUJOCO_KEY_BUNDLE') or os.path.exists(os.path.expanduser('~/.mujoco')))
    if skip_mujoco and spec._entry_point.startswith('gym.envs.mujoco:'):
        return

    with helpers.tempdir() as temp:
        env = spec.make()
        # Skip un-renderable envs
        if 'human' not in env.metadata.get('render.modes', []):
            return

        env.monitor.start(temp)
        env.reset()
        env.monitor.close()

        env.reset()
        env.render()
        env.render(close=True)

        env.close()
