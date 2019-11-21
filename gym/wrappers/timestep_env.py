from enum import IntEnum
from dataclasses import dataclass

import gym


class StepType(IntEnum):
    FIRST = 0
    MID = 1
    LAST = 2


@dataclass
class TimeStep:
    step_type: StepType
    observation: object
    reward: float
    done: bool
    info: dict

    def __getitem__(self, key):
        return self.info[key]

    def first(self):
        if self.step_type == StepType.FIRST:
            assert all([x is None for x in [self.reward, self.done, self.info]])
        return self.step_type == StepType.FIRST

    def mid(self):
        if self.step_type == StepType.MID:
            assert not self.first() and not self.last()
        return self.step_type == StepType.MID

    def last(self):
        if self.step_type == StepType.LAST:
            assert self.done is not None and self.done
        return self.step_type == StepType.LAST

    def time_limit(self):
        return self.last() and self.info.get('TimeLimit.truncated', False)

    def terminal(self):
        return self.last() and not self.time_limit()

    def __repr__(self):
        return f'{self.__class__.__name__}({self.step_type.name})'


class TimeStepEnv(gym.Wrapper):
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        step_type = StepType.LAST if done else StepType.MID
        timestep = TimeStep(step_type=step_type, observation=observation, reward=reward, done=done, info=info)
        return timestep

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        return TimeStep(StepType.FIRST, observation=observation, reward=None, done=None, info=None)
