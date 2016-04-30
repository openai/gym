import atexit
import logging
import json
import numpy as np
import os
import six
import sys
import threading
import weakref

from gym import error, version
from gym.monitoring import stats_recorder, video_recorder

logger = logging.getLogger(__name__)

FILE_PREFIX = 'openaigym'
MANIFEST_PREFIX = FILE_PREFIX + '.manifest'

i = -1
lock = threading.Lock()
def next_monitor_id():
    global i
    with lock:
        i += 1
        return i

def detect_training_manifests(training_dir):
    return [os.path.join(training_dir, f) for f in os.listdir(training_dir) if f.startswith(MANIFEST_PREFIX + '.')]

def detect_monitor_files(training_dir):
    return [os.path.join(training_dir, f) for f in os.listdir(training_dir) if f.startswith(FILE_PREFIX + '.')]

def clear_monitor_files(training_dir):
    files = detect_monitor_files(training_dir)
    if len(files) == 0:
        return

    logger.info('Clearing %d monitor files from previous run (because force=True was provided)', len(files))
    for file in files:
        os.unlink(file)

def capped_cubic_video_schedule(episode_id):
    if episode_id < 1000:
        return int(round(episode_id ** (1. / 3))) ** 3 == episode_id
    else:
        return episode_id % 1000 == 0

# Monitors will automatically close themselves when garbage collected
# (via __del__) or when the program exits (via close_all_monitors's
# atexit behavior).
monitors = weakref.WeakValueDictionary()
def ensure_close_at_exit(monitor):
    monitors[monitor.monitor_id] = monitor

@atexit.register
def close_all_monitors():
    for key, monitor in monitors.items():
        monitor.close()

class Monitor(object):
    """A configurable monitor for your training runs.

    Every env has an attached monitor, which you can access as
    'env.monitor'. Simple usage is just to call 'monitor.start(dir)'
    to begin monitoring and 'monitor.close()' when training is
    complete. This will record stats and will periodically record a video.

    For finer-grained control over how often videos are collected, use the
    video_callable argument, e.g.
    'monitor.start(video_callable=lambda count: count % 100 == 0)'
    to record every 100 episodes. ('count' is how many episodes have completed)

    Depending on the environment, video can slow down execution. You
    can also use 'monitor.configure(video_callable=lambda count: False)' to disable
    video.

    Monitor supports multiple threads and multiple processes writing
    to the same directory of training data. The data will later be
    joined by scoreboard.upload_training_data and on the server.

    Args:
        env (gym.Env): The environment instance to monitor.

    Attributes:
        id (Optional[str]): The ID of the monitored environment

    """

    def __init__(self, env):
        self.env = env
        self.videos = []

        self.stats_recorder = None
        self.video_recorder = None
        self.enabled = False
        self.episode_id = 0

        self.monitor_id = next_monitor_id()

        ensure_close_at_exit(self)

    def start(self, directory, video_callable=None, force=False):
        """Start monitoring.

        Args:
            directory (str): A per-training run directory where to record stats.
            video_callable: function that takes in the index of the episode and outputs a boolean, indicating whether we should record a video on this episode. The default is to take perfect cubes.
            force (bool): Clear out existing training data from this directory (by deleting every file prefixed with "openaigym.").
        """
        if self.env.spec is None:
            logger.warn("Trying to monitor an environment which has no 'spec' set. This usually means you did not create it via 'gym.make', and is recommended only for advanced users.")

        if not os.path.exists(directory):
            logger.info('Creating monitor directory %s', directory)
            os.makedirs(directory)

        if video_callable is None:
            video_callable = capped_cubic_video_schedule

        # Check on whether we need to clear anything
        if force:
            clear_monitor_files(directory)
        else:
            training_manifests = detect_training_manifests(directory)
            if len(training_manifests) > 0:
                raise error.Error('''Trying to write to monitor directory {} with existing monitor files: {}.

 You should use a unique directory for each training run, or use 'force=True' to automatically clear previous monitor files.'''.format(directory, ', '.join(training_manifests[:5])))


        self.enabled = True
        self.directory = os.path.abspath(directory)
        # We use the 'openai-gym' prefix to determine if a file is
        # ours
        self.file_prefix = FILE_PREFIX
        self.file_infix = str(self.monitor_id)
        self.stats_recorder = stats_recorder.StatsRecorder(directory, '{}.episode_batch.{}'.format(self.file_prefix, self.file_infix))
        self.configure(video_callable=video_callable)
        if not os.path.exists(directory):
            os.mkdir(directory)

    def close(self):
        """Flush all monitor data to disk and close any open rending windows."""
        if not self.enabled:
            return
        stats_file = None

        if self.stats_recorder:
            stats_file = self.stats_recorder.close()
        if self.video_recorder is not None:
            self._close_video_recorder()
        # Note we'll close the env's rendering window even if we did
        # not open it. There isn't a particular great way to know if
        # we did, since some environments will have a window pop up
        # during video recording.
        try:
            self.env.render(close=True)
        except Exception as e:
            if self.env.spec:
                key = self.env.spec.id
            else:
                key = self.env
            # We don't want to avoid writing the manifest simply
            # because we couldn't close the renderer.
            logger.error('Could not close renderer for %s: %s', key, e)

        # Give it a very distiguished name, since we need to pick it
        # up from the filesystem later.
        path = os.path.join(self.directory, '{}.manifest.{}.{}.manifest.json'.format(self.file_prefix, self.file_infix, os.getpid()))
        logger.debug('Writing training manifest file to %s', path)
        with open(path, 'w') as f:
            # We need to write relative paths here since people may
            # move the training_dir around. It would be cleaner to
            # already have the basenames rather than basename'ing
            # manually, but this works for now.
            json.dump({
                'stats': os.path.basename(stats_file),
                'videos': [(os.path.basename(v), os.path.basename(m))
                           for v, m in self.videos],
                'env_info': self._env_info(),
            }, f)
        self.enabled = False
        # Stop tracking this for autoclose
        del monitors[self.monitor_id]

        logger.info('''Finished writing results. You can upload them to the scoreboard via gym.upload(%r)''', self.directory)

    def configure(self, video_callable=None):
        """Reconfigure the monitor.

            video_callable (function): Whether to record video to upload to the scoreboard.
        """
        if video_callable is not None:
            self.video_callable = video_callable

    def _before_step(self, action):
        if not self.enabled: return
        self.stats_recorder.before_step(action)

    def _after_step(self, observation, reward, done, info):
        if not self.enabled: return done

        # Add 1 since about to take another step
        if self.env.spec and self.stats_recorder.steps+1 >= self.env.spec.timestep_limit:
            logger.info('Ending episode %i because it reached the timestep limit of %i.', self.episode_id, self.env.spec.timestep_limit)
            done = True

        # Record stats
        self.stats_recorder.after_step(observation, reward, done, info)
        # Record video
        self.video_recorder.capture_frame()

        return done


    def _before_reset(self):
        if not self.enabled: return
        self.stats_recorder.before_reset()

    def _after_reset(self, observation):
        if not self.enabled: return

        # Reset the stat count
        self.stats_recorder.after_reset(observation)

        # Close any existing video recorder
        if self.video_recorder:
            self._close_video_recorder()

        # Start recording the next video.
        self.video_recorder = video_recorder.VideoRecorder(
            env=self.env,
            base_path=os.path.join(self.directory, '{}.video.{}.{}.video{:06}'.format(self.file_prefix, self.file_infix, os.getpid(), self.episode_id)),
            metadata={'episode_id': self.episode_id},
            enabled=self._video_enabled(),
        )
        self.video_recorder.capture_frame()

        # Bump *after* all reset activity has finished
        self.episode_id += 1

    def _close_video_recorder(self):
        self.video_recorder.close()
        if self.video_recorder.functional:
            self.videos.append((self.video_recorder.path, self.video_recorder.metadata_path))

    def _video_enabled(self):
        return self.video_callable(self.episode_id)

    def _env_info(self):
        if self.env.spec:
            return {
                'env_id': self.env.spec.id,
                'gym_version': version.VERSION,
            }
        else:
            return {}

    def __del__(self):
        # Make sure we've closed up shop when garbage collecting
        self.close()

def load_results(training_dir):
    if not os.path.exists(training_dir):
        return

    manifests = detect_training_manifests(training_dir)
    if not manifests:
        return

    logger.debug('Uploading data from manifest %s', ', '.join(manifests))

    # Load up stats + video files
    stats_files = []
    videos = []
    env_infos = []

    for manifest in manifests:
        with open(manifest) as f:
            contents = json.load(f)
            # Make these paths absolute again
            stats_files.append(os.path.join(training_dir, contents['stats']))
            videos += [(os.path.join(training_dir, v), os.path.join(training_dir, m))
                       for v, m in contents['videos']]
            env_infos.append(contents['env_info'])

    env_info = collapse_env_infos(env_infos, training_dir)
    timestamps, episode_lengths, episode_rewards, initial_reset_timestamp = merge_stats_files(stats_files)

    return {
        'manifests': manifests,
        'env_info': env_info,
        'timestamps': timestamps,
        'episode_lengths': episode_lengths,
        'episode_rewards': episode_rewards,
        'initial_reset_timestamp': initial_reset_timestamp,
        'videos': videos,
    }

def merge_stats_files(stats_files):
    timestamps = []
    episode_lengths = []
    episode_rewards = []
    initial_reset_timestamps = []

    for path in stats_files:
        with open(path) as f:
            content = json.load(f)
            timestamps += content['timestamps']
            episode_lengths += content['episode_lengths']
            episode_rewards += content['episode_rewards']
            initial_reset_timestamps.append(content['initial_reset_timestamp'])

    idxs = np.argsort(timestamps)
    timestamps = np.array(timestamps)[idxs].tolist()
    episode_lengths = np.array(episode_lengths)[idxs].tolist()
    episode_rewards = np.array(episode_rewards)[idxs].tolist()
    initial_reset_timestamp = min(initial_reset_timestamps)
    return timestamps, episode_lengths, episode_rewards, initial_reset_timestamp

def collapse_env_infos(env_infos, training_dir):
    assert len(env_infos) > 0

    first = env_infos[0]
    for other in env_infos[1:]:
        if first != other:
            raise error.Error('Found two unequal env_infos: {} and {}. This usually indicates that your training directory {} has commingled results from multiple runs.'.format(first, other, training_dir))

    for key in ['env_id', 'gym_version']:
        if key not in first:
            raise error.Error("env_info {} from training directory {} is missing expected key {}. This is unexpected and likely indicates a bug in gym.".format(first, training_dir, key))
    return first
