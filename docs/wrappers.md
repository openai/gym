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

## Action Wrappers

`ClipAction(env)` [text]
* Needs review (including for good assertion messages and test coverage)

`ChangeDtype(env, dtype)` [text]
* Bring over from SuperSuit

`RescaleAction(env, min_action, max_action)` [text]
* Needs review (including for good assertion messages and test coverage)

`StickyActions(env, repeat_action_probability)` [text]
* Create as seperate wrapper from Atari/bring over from SuperSuit

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

`RecordVideo(env, ...)` [text]
* https://github.com/openai/gym/pull/2300

`TimeLimit(env, max_episode_steps)` [text]
* Needs review (including for good assertion messages and test coverage)

Some sort of vector environment conversion wrapper needs to be added here, this will be figured out after the API is changed.
