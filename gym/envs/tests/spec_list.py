from gym import envs
import os
import logging
logger = logging.getLogger(__name__)

def should_skip_env_spec_for_tests(spec):
    # We skip tests for envs that require dependencies or are otherwise
    # troublesome to run frequently
    ep = spec._entry_point
    # Skip mujoco tests for pull request CI
    skip_mujoco = not (os.environ.get('MUJOCO_KEY_BUNDLE') or os.path.exists(os.path.expanduser('~/.mujoco')))
    if skip_mujoco and ep.startswith('gym.envs.mujoco:'):
        return True
    if (    spec.id.startswith("Go") or 
            spec.id.startswith("Hex") or 
            ep.startswith('gym.envs.box2d:') or 
            ep.startswith('gym.envs.parameter_tuning:') or 
            ep.startswith('gym.envs.safety:Semisuper') or
            (ep.startswith("gym.envs.atari") and not spec.id.startswith("Pong"))
    ):
        logger.warning("Skipping tests for env {}".format(ep))
        return True
    return False

spec_list = [spec for spec in sorted(envs.registry.all(), key=lambda x: x.id) if spec._entry_point is not None and not should_skip_env_spec_for_tests(spec)]
