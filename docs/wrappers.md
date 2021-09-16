Gym includes numerous wrappers for environments that include preprocessing and various other functions. The following categories are included:

## Observation Wrappers

`DelayObservation(env, delay)` [text]
* Bring over from SuperSuit

`FilterObservation(env, filter_keys)` [text]
* Needs review (including for good assertion messages and test coverage)

`FlattenObservation(env)` [text]
* Needs good asseration messages

`FrameStack(env, num_stack, lz4_compress=False)` [text]
* Needs review (including for good assertion messages and test coverage)

`GrayScaleObservation(env)` [text] 
* Needs R/G/B channel argument added like supersuit wrapper
* Needs review (including for good assertion messages and test coverage)
* Needs CV2 dependency replaced with SuperSuit's way of doing full grey scaling: https://github.com/PettingZoo-Team/SuperSuit/blob/master/supersuit/utils/basic_transforms/color_reduction.py

`PixelObservationWrapper(pixels_only=True, render_kwargs=None, pixel_keys=("pixels",))` [text]
* Needs review (including for good assertion messages and test coverage)

`RescaleObservation(env, min_obs, max_obs`) [text]
* Bring over from Supersuit or from https://github.com/openai/gym/pull/1635

`ResizeObservation(env, shape)` [text]
* Needs review (including for good assertion messages and test coverage)
* Switch from CV2 to Lycon2 once released

`TimeAwareObservation(env)` [text]
* Needs review (including for good assertion messages and test coverage)


`NormalizeObservation(env, epsilon=1e-8)` [text]
* This wrapper normalizes the observations to have approximately zero mean and unit variance


`NormalizeReward(env, gamma=0.99, epsilon=1e-8)` [text]
* This wrapper scales the rewards, which are divided through by the standard deviation of a rolling discounted returns. See page 3 of from [Engstrom, Ilyas et al. (2020)](https://arxiv.org/pdf/2005.12729.pdf)

## Action Wrappers

`ClipAction(env)` [text]
* Needs review (including for good assertion messages and test coverage)

`ChangeDtype(env, dtype)` [text]
* Bring over from SuperSuit

`RescaleAction(env, min_action, max_action)` [text]
* Needs review (including for good assertion messages and test coverage)

`StickyActions(env, repeat_action_probability)` [text]
* Create as separate wrapper from Atari/bring over from SuperSuit

## Reward Wrappers

`ClipReward(env, lower_bound-1, upper_bound=1)` [text]
* Bring over from SuperSuit

## Lambda Wrappers

`ActionLambda(env, change_action_fn, change_space_fn)` [text]
* Bring over from SuperSuit

`ObservationLambda(env, observation_fn, observation_space_fn)` [text]
* Bring over from SuperSuit, replaces TransformObservation

`RewardLambda(env, reward_fn)` [text]
* Bring over from SuperSuit, replaces TransformReward

## Other
`AtariPreprocessing(env, noop_max=30, frame_skip=4, screen_size=84, terminal_on_life_loss=False, grayscale_obs=True, grayscale_newaxis=False, scale_obs=False)` [text]
* Needs review (including for good assertion messages and test coverage)

`RecordEpisodeStatistic(env)` [text]
* Needs review (including for good assertion messages and test coverage)

`RecordVideo(env, video_folder, episode_trigger, step_trigger, video_length=0, name_prefix="rl-video")` [text]

The `RecordVideo` is a lightweight `gym.Wrapper` that helps recording videos. See the following
code as an example.

```python
import gym
from gym.wrappers import RecordVideo, capped_cubic_video_schedule
env = gym.make("CartPole-v1")
env = RecordVideo(env, "videos")
# the above is equivalent as
# env = RecordVideo(env, "videos", episode_trigger=capped_cubic_video_schedule)
observation = env.reset()
for _ in range(1000):
  env.render()
  action = env.action_space.sample() # your agent here (this takes random actions)
  observation, reward, done, info = env.step(action)

  if done:
    observation = env.reset()
env.close()
```

To use it, you need to specify the `video_folder` as the storing location. By default
the `RecordVideo` uses episode counts to trigger video recording based on the `episode_trigger=capped_cubic_video_schedule`,
which is a cubic progression for early episodes (1,8,27,...) and then every 1000 episodes (1000, 2000, 3000...).
This can be changed by modifying the `episode_trigger` argument of the `RecordVideo`).

Alternatively, you may also trigger the the video recording based on the environment steps via the  `step_trigger` like

```python
import gym
from gym.wrappers import RecordVideo
env = gym.make("CartPole-v1")
env = RecordVideo(env, "videos", step_trigger=lambda x: x % 100 == 0)
observation = env.reset()
for _ in range(1000):
  env.render()
  action = env.action_space.sample() # your agent here (this takes random actions)
  observation, reward, done, info = env.step(action)

  if done:
    observation = env.reset()
env.close()
```

Which will trigger the video recording at exactly every 100 environment steps (unless the previous recording hasn't finished yet).

Note that you may use exactly one trigger (i.e. `step_trigger` or `record_video_trigger`) at a time.

There are two modes to end the video recording:
1. Episodic mode. 
    * By default `video_length=0` means the wrapper will record *episodic* videos: it will keep
    record the frames until the env returns `done=True`.
2. Fixed-interval mode.
    * By tuning `video_length` such as `video_length=100`, the wrapper will record exactly 100 frames
    for every videos the wrapper creates. 

Lastly the `name_prefix` allows you to customize the name of the videos.


`TimeLimit(env, max_episode_steps)` [text]
* Needs review (including for good assertion messages and test coverage)

`OrderEnforcing(env)` [text]

`OrderEnforcing` is a light-weight wrapper that throws an exception when `env.step()` is called before `env.reset()`, the wrapper is enabled by default for environment specs without `max_episode_steps` and can be disabled by passing `order_enforce=False` like:
```python3
register(
    id="CustomEnv-v1",
    entry_point="...",
    order_enforce=False,
)
```

Some sort of vector environment conversion wrapper needs to be added here, this will be figured out after the API is changed.
