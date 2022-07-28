from typing import Callable, Optional
import gym

try:
    import moviepy
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
    ):

    assert isinstance(frames, list), f"Expected a list of frames, got a {frames.__class__.__name__} instead."
