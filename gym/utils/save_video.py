"""Utility functions to save rendering videos."""
import os
from typing import Callable, Optional

import gym
from gym import logger

try:
    from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
except ImportError:
    raise gym.error.DependencyNotInstalled(
        "MoviePy is not installed, run `pip install moviepy`"
    )


def capped_cubic_video_schedule(episode_id: int) -> bool:
    """The default episode trigger.

    This function will trigger recordings at the episode indices 0, 1, 4, 8, 27, ..., :math:`k^3`, ..., 729, 1000, 2000, 3000, ...

    Args:
        episode_id: The episode number

    Returns:
        If to apply a video schedule number
    """
    if episode_id < 1000:
        return int(round(episode_id ** (1.0 / 3))) ** 3 == episode_id
    else:
        return episode_id % 1000 == 0


def save_video(
    frames: list,
    video_folder: str,
    episode_trigger: Callable[[int], bool] = None,
    step_trigger: Callable[[int], bool] = None,
    video_length: Optional[int] = None,
    name_prefix: str = "rl-video",
    episode_index: int = 0,
    step_starting_index: int = 0,
    **kwargs,
):
    """Save videos from rendering frames.

    This function extract video from a list of render frame episodes.

    Args:
        frames (List[RenderFrame]): A list of frames to compose the video.
        video_folder (str): The folder where the recordings will be stored
        episode_trigger: Function that accepts an integer and returns ``True`` iff a recording should be started at this episode
        step_trigger: Function that accepts an integer and returns ``True`` iff a recording should be started at this step
        video_length (int): The length of recorded episodes. If it isn't specified, the entire episode is recorded.
            Otherwise, snippets of the specified length are captured.
        name_prefix (str): Will be prepended to the filename of the recordings.
        episode_index (int): The index of the current episode.
        step_starting_index (int): The step index of the first frame.
        **kwargs: The kwargs that will be passed to moviepy's ImageSequenceClip.
            You need to specify either fps or duration.

    Example:
        >>> import gym
        >>> from gym.utils.save_video import save_video
        >>> env = gym.make("FrozenLake-v1", render_mode="rgb_array_list")
        >>> env.reset()
        >>> step_starting_index = 0
        >>> episode_index = 0
        >>> for step_index in range(199):
        ...    action = env.action_space.sample()
        ...    _, _, done, _ = env.step(action)
        ...    if done:
        ...       save_video(
        ...          env.render(),
        ...          "videos",
        ...          fps=env.metadata["render_fps"],
        ...          step_starting_index=step_starting_index,
        ...          episode_index=episode_index
        ...       )
        ...       step_starting_index = step_index + 1
        ...       episode_index += 1
        ...       env.reset()
        >>> env.close()
    """
    if not isinstance(frames, list):
        logger.error(f"Expected a list of frames, got a {type(frames)} instead.")
    if episode_trigger is None and step_trigger is None:
        episode_trigger = capped_cubic_video_schedule

    video_folder = os.path.abspath(video_folder)
    os.makedirs(video_folder, exist_ok=True)
    path_prefix = f"{video_folder}/{name_prefix}"

    if episode_trigger is not None and episode_trigger(episode_index):
        clip = ImageSequenceClip(frames[:video_length], **kwargs)
        clip.write_videofile(f"{path_prefix}-episode-{episode_index}.mp4")

    if step_trigger is not None:
        # skip the first frame since it comes from reset
        for step_index, frame_index in enumerate(
            range(1, len(frames)), start=step_starting_index
        ):
            if step_trigger(step_index):
                end_index = (
                    frame_index + video_length if video_length is not None else None
                )
                clip = ImageSequenceClip(frames[frame_index:end_index], **kwargs)
                clip.write_videofile(f"{path_prefix}-step-{step_index}.mp4")
