from typing import Optional, Tuple, Union

import pytest

import gym
from gym import spaces
from gym.core import ObsType
from gym.envs.registration import EnvSpec
from gym.spaces import Discrete
from gym.utils import seeding
from gym.utils.env_checker import (
    check_env,
    check_render,
    check_reset_info,
    check_reset_options,
    check_reset_seed,
)


class TestingMinimalEnv(gym.Env):
    def __init__(
        self,
        render_modes=None,
        render_fps=None,
        render_mode=None,
        action_space=Discrete(2),
        observation_space=Discrete(2),
    ):
        self.metadata["render_modes"] = render_modes
        self.metadata["render_fps"] = render_fps
        self.render_mode = render_mode

        if observation_space:
            self.observation_space = observation_space
        if action_space:
            self.action_space = action_space

    def reset(self):
        raise NotImplementedError("Minimal env reset should not be run.")


class TestingResetEnv(gym.Env):
    def __init__(self, reset_fn, obs_space=spaces.Box(0, 1, ())):
        self.observation_space = obs_space
        self.reset_fn = reset_fn

        self.spec = EnvSpec("TestingResetEnv-v0")

    def reset(
        self,
        *,
        seed: Optional[int] = "Weird default value",
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> Union[ObsType, Tuple[ObsType, dict]]:
        return self.reset_fn(self, seed, return_info, options)


def test_check_reset_seed():
    """Tests:
    * no reset seed parameter
    * outside observation space
    * reset non deterministic observations
    * no super.reset
    * default seed parameter is not none
    """
    with pytest.raises(
        gym.error.Error,
        match="The `reset` method does not provide the `seed` keyword argument",
    ):
        check_reset_seed(TestingMinimalEnv())

    with pytest.raises(
        AssertionError,
        match=r"The observation returns by `env\.reset\(seed=123\)` is not within the observation space",
    ):
        check_reset_seed(
            TestingResetEnv(
                lambda self, _, __, ___: self.observation_space.sample()
                + self.observation_space.high
            )
        )

    with pytest.raises(
        AssertionError,
        match=r"`env\.reset\(seed=123\)` is not deterministic as the observations are not equivalent",
    ):
        check_reset_seed(
            TestingResetEnv(lambda self, _, __, ___: self.observation_space.sample())
        )

    with pytest.raises(
        AssertionError,
        match=r"Mostly likely the environment reset function does not call `super\(\)\.reset\(seed=seed\)` "
        r"as the random generates are not same when the same seeds are passed to `env\.reset`\.",
    ):

        def _no_super_reset(self, seed, _, __):
            self.np_random.random()  # generates a new prng
            return seeding.np_random(seed)[
                0
            ].random()  # generate seed deterministic result

        check_reset_seed(TestingResetEnv(_no_super_reset))

    with pytest.raises(
        AssertionError,
        match=r"Mostly likely the environment reset function does not call `super\(\)\.reset\(seed=seed\)` "
        r"as the random generates are not different when different seeds are passed to `env\.reset`\.",
    ):

        def _super_reset_fixed(self, _, __, ___):
            super(TestingResetEnv, self).reset(
                seed=1
            )  # Call super that ignores the seed passed
            return self.np_random.random()  # Deterministic output

        check_reset_seed(TestingResetEnv(_super_reset_fixed))

    with pytest.warns(
        UserWarning,
        match="The default seed argument in reset should be `None`, "
        "otherwise the environment will by default always be deterministic",
    ):

        def valid_reset(self, seed, _, __):
            super(TestingResetEnv, self).reset(seed=seed)
            return self.np_random.random()

        check_reset_seed(TestingResetEnv(valid_reset))

    # Check that function runs normally
    with pytest.warns(None):
        check_reset_seed(gym.make("CartPole-v1", disable_env_checker=True))


def test_check_reset_info():
    """Tests:
    * no reset info parameter
    * Number of return values not 1 or 2
    * Info type not dict
    """

    with pytest.raises(
        gym.error.Error,
        match="The `reset` method does not provide the `return_info` keyword argument",
    ):
        check_reset_info(TestingMinimalEnv())

    with pytest.raises(
        AssertionError,
        match=r"The value returned by `env\.reset\(return_info=True\)` is not within the observation space",
    ):
        check_reset_info(
            TestingResetEnv(
                lambda self, _, __, ___: self.observation_space.sample()
                + self.observation_space.high
            )
        )

    with pytest.raises(
        AssertionError,
        match="Calling the reset method with `return_info=True` did not return a 2-tuple",
    ):
        check_reset_info(
            TestingResetEnv(
                lambda self, _, return_info, ___: [1, 2, 3] if return_info else 0
            )
        )

    with pytest.raises(
        AssertionError,
        match=r"The second element returned by `env\.reset\(return_info=True\)` is not within the observation space",
    ):
        check_reset_info(
            TestingResetEnv(
                lambda self, _, return_info, ___: (
                    self.observation_space.sample() + self.observation_space.high,
                    {},
                )
                if return_info
                else 0
            )
        )

    with pytest.raises(
        AssertionError,
        match=r"The second element returned by `env\.reset\(return_info=True\)` was not a dictionary",
    ):
        check_reset_info(
            TestingResetEnv(
                lambda self, _, return_info, ___: (0, ["key", "value"])
                if return_info
                else 0
            )
        )

    with pytest.warns(None):
        check_reset_info(gym.make("CartPole-v1", disable_env_checker=True))


def test_check_reset_options():
    """Tests:
    * no reset options
    """

    with pytest.raises(
        gym.error.Error,
        match="The `reset` method does not provide the `options` keyword argument",
    ):
        check_reset_options(TestingMinimalEnv())

    with pytest.warns(None):
        check_reset_options(gym.make("CartPole-v1", disable_env_checker=True))


def test_check_render():
    """Tests:
    * No render modes
    * No render fps
    * No render mode attribute
    * Incorrect render mode to render modes
    """
    with pytest.raises(
        gym.error.Error,
        match="No render modes was declared in the environment "
        r"\(env\.metadata\['render_modes'\] is None or not defined\), "
        r"you may have trouble when calling `\.render\(\)`\.",
    ):
        check_render(TestingMinimalEnv(render_modes=None))

    with pytest.raises(
        AssertionError,
        match=r"Expects the render_modes to be a sequence \(i\.e\. list, tuple\), actual type: <class 'str'>",
    ):
        check_render(TestingMinimalEnv(render_modes="Testing mode"))

    with pytest.raises(
        AssertionError,
        match=r"Expects all render modes to be strings, actual types: \[<class 'str'>, <class 'int'>\]\.",
    ):
        check_render(TestingMinimalEnv(render_modes=["Testing mode", 1]))

    with pytest.warns(
        UserWarning,
        match="No render fps was declared in the environment "
        r"\(env.metadata\['render_fps'\] is None or not defined\), "
        r"rendering may occur at inconsistent fps\.",
    ):
        check_render(
            TestingMinimalEnv(
                render_modes=["Testing mode"],
                render_fps=None,
                render_mode="Testing mode",
            )
        )

    with pytest.raises(
        AssertionError,
        match=r"Expects the `env.metadata\['render_fps'\]` to be an integer, actual type: <class 'str'>\.",
    ):
        check_render(TestingMinimalEnv(render_modes=["Testing mode"], render_fps="fps"))

    with pytest.raises(
        AssertionError, match="With no render_modes, expects the render_mode to be None"
    ):
        check_render(
            TestingMinimalEnv(render_modes=[], render_fps=30, render_mode="Test")
        )

    with pytest.raises(
        AssertionError,
        match=r"The environment was initialized successfully however with an unsupported render mode\.",
    ):
        check_render(
            TestingMinimalEnv(
                render_modes=["Testing mode"], render_fps=30, render_mode="Non mode"
            )
        )


def test_check_env():
    """Tests:
    * env is not an isinstance of gym.Env
    * no action space
    * no observation space

    The rest of the env_checker if tested in their individual respective test functions
    """

    with pytest.raises(
        AssertionError,
        match="Your environment must inherit from the gym.Env class https://www.gymlibrary.ml/content/environment_creation/",
    ):
        check_env(env=["imaginary environment"])

    with pytest.raises(
        AssertionError,
        match="You must specify a action space. https://www.gymlibrary.ml/content/environment_creation/",
    ):
        check_env(TestingMinimalEnv(action_space=None))

    with pytest.raises(
        AssertionError,
        match="You must specify an observation space. https://www.gymlibrary.ml/content/environment_creation/",
    ):
        check_env(TestingMinimalEnv(observation_space=None))
