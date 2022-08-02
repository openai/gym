"""Utility functions to save rendering videos."""
import os
from typing import Callable

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
    video_length: int = -1,
    name_prefix: str = "rl-video",
    episode_starting_index: int = 0,
    step_starting_index: int = 0,
    **kwargs,
):
    """Save videos from rendering frames.

    This function extract video from a list of render frame episodes.

    Args:
        frames (List[List[RenderFrame]] | List[RenderFrame]): A list of frames to compose the video.
        In case of single episode, the expected input is a List[RenderFrame].
        In case of multiple episode, the expected input is a List[List[RenderFrame]].
        video_folder (str): The folder where the recordings will be stored
        episode_trigger: Function that accepts an integer and returns ``True`` iff a recording should be started at this episode
        step_trigger: Function that accepts an integer and returns ``True`` iff a recording should be started at this step
        video_length (int): The length of recorded episodes. If -1, entire episodes are recorded.
            Otherwise, snippets of the specified length are captured.
        name_prefix (str): Will be prepended to the filename of the recordings.
        episode_starting_index (int): The index of the first episode in frames.
        step_starting_index (int): The step index of the first frame.
        fps (float): Frame per second of the video.
    """
    if not isinstance(frames, list):
        logger.error(
            f"Expected a list of frames, got a {frames.__class__.__name__} instead."
        )
    if len(frames) == 0:
        return
    if not isinstance(frames[0], list):
        frames = [frames]

    if episode_trigger is None and step_trigger is None:
        episode_trigger = capped_cubic_video_schedule

    video_folder = os.path.abspath(video_folder)
    if os.path.isdir(video_folder):
        logger.warn(
            f"Overwriting existing videos at {video_folder} folder "
            f"(try specifying a different `video_folder` if this is not desired)"
        )
    os.makedirs(video_folder, exist_ok=True)
    path_prefix = f"{video_folder}/{name_prefix}"

    step_index = step_starting_index
    for episode_index, episode in enumerate(frames, start=episode_starting_index):

        if episode_trigger is not None and episode_trigger(episode_index):
            clip = ImageSequenceClip(episode[:video_length], **kwargs)
            clip.write_videofile(f"{path_prefix}-episode-{episode_index}.mp4")

        if step_trigger is not None:
            # skip the first frame since it comes from reset
            for frame_index in range(1, len(episode)):
                if step_trigger(step_index):
                    end_index = frame_index + video_length if video_length > 0 else -1
                    clip = ImageSequenceClip(episode[frame_index:end_index], **kwargs)
                    clip.write_videofile(f"{path_prefix}-step-{step_index}.mp4")

                step_index += 1
