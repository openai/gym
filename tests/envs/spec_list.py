from gym import envs, logger

SKIP_MUJOCO_V3_WARNING_MESSAGE = (
    "Cannot run mujoco test because `mujoco-py` is not installed"
)
SKIP_MUJOCO_V4_WARNING_MESSAGE = (
    "Cannot run mujoco test because `mujoco` is not installed"
)

skip_mujoco_v3 = False
try:
    import mujoco_py  # noqa:F401
except ImportError:
    skip_mujoco_v3 = True

skip_mujoco_v4 = False
try:
    import mujoco  # noqa:F401
except ImportError:
    skip_mujoco_v4 = True


def should_skip_env_spec_for_tests(spec):
    # We skip tests for envs that require dependencies or are otherwise
    # troublesome to run frequently
    ep = spec.entry_point
    # Skip mujoco tests for pull request CI
    if (skip_mujoco_v3 or skip_mujoco_v4) and ep.startswith("gym.envs.mujoco"):
        return True
    try:
        import gym.envs.atari  # noqa:F401
    except ImportError:
        if ep.startswith("gym.envs.atari"):
            return True
    try:
        import Box2D  # noqa:F401
    except ImportError:
        if ep.startswith("gym.envs.box2d"):
            return True

    if (
        "GoEnv" in ep
        or "HexEnv" in ep
        or (
            ep.startswith("gym.envs.atari")
            and not spec.id.startswith("Pong")
            and not spec.id.startswith("Seaquest")
        )
    ):
        logger.warn(f"Skipping tests for env {ep}")
        return True
    return False


def skip_mujoco_py_env_for_test(spec):
    ep = spec.entry_point
    version = spec.version
    if ep.startswith("gym.envs.mujoco") and version < 4:
        return True
    return False


spec_list = [
    spec
    for spec in sorted(envs.registry.values(), key=lambda x: x.id)
    if spec.entry_point is not None and not should_skip_env_spec_for_tests(spec)
]
spec_list_no_mujoco_py = [
    spec for spec in spec_list if not skip_mujoco_py_env_for_test(spec)
]
