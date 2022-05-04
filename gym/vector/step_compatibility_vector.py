from gym.vector.vector_env import VectorEnvWrapper
from gym.wrappers.step_compatibility import step_to_new_api, step_to_old_api


def step_api_vector_compatibility(VectorEnvClass):
    class StepCompatibilityVector(VectorEnvWrapper):
        r"""A wrapper which can transform a vector environment to a new or old step API.

        Old step API refers to step() method returning (observation, reward, done, info)
        New step API refers to step() method returning (observation, reward, terminated, truncated, info)
        (Refer to docs for details on the API change)

        This wrapper is to be used to ease transition to new API. It will be removed in v1.0

        Parameters
        ----------
            env (gym.vector.VectorEnv): the vector env to wrap. Has to be in new step API
            new_step_api (bool): True to use vector env with new step API, False to use vector env with old step API. (True by default)

        """

        def __init__(self, *args, **kwargs):
            self.new_step_api = kwargs.get("new_step_api", False)
            kwargs.pop("new_step_api", None)
            super().__init__(VectorEnvClass(*args, **kwargs))

        def step_wait(self):
            step_returns = self.env.step_wait()
            if self.new_step_api:
                return step_to_new_api(step_returns)
            else:
                return step_to_old_api(step_returns)

        def __del__(self):
            self.env.__del__()

    return StepCompatibilityVector
