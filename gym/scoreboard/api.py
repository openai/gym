import logging
import json
import os
import re
import tarfile
import tempfile
from gym import error, monitoring
from gym.scoreboard.client import resource, util

MAX_VIDEOS = 100

logger = logging.getLogger(__name__)

video_name_re = re.compile('^[\w.-]+\.(mp4|avi|json)$')
metadata_name_re = re.compile('^[\w.-]+\.meta\.json$')

def upload(training_dir, algorithm_id=None, writeup=None, api_key=None):
    """Upload the results of training (as automatically recorded by your
    env's monitor) to OpenAI Gym.

    Args:
        training_dir (Optional[str]): A directory containing the results of a training run.
        algorithm_id (Optional[str]): An arbitrary string indicating the paricular version of the algorithm (including choices of parameters) you are running.
        writeup (Optional[str]): A Gist URL (of the form https://gist.github.com/<user>/<id>) containing your writeup for this evaluation.
        api_key (Optional[str]): Your OpenAI API key. Can also be provided as an environment variable (OPENAI_GYM_API_KEY).
    """

    open_monitors = monitoring._monitors.values()
    if open_monitors:
        envs = [m.env.spec.id if m.env.spec else '(unknown)' for m in open_monitors]
        raise error.Error("Still have an open monitor on {}. You must run 'env.monitor.close()' before uploading.".format(', '.join(envs)))

    env_info, training_episode_batch, training_video = upload_training_data(training_dir, api_key=api_key)
    env_id = env_info['env_id']
    training_episode_batch_id = training_video_id = None
    if training_episode_batch:
        training_episode_batch_id = training_episode_batch.id
    if training_video:
        training_video_id = training_video.id

    if logger.level <= logging.INFO:
        if training_episode_batch_id is not None and training_video_id is not None:
            logger.info('[%s] Creating evaluation object from %s with learning curve and training video', env_id, training_dir)
        elif training_episode_batch_id is not None:
            logger.info('[%s] Creating evaluation object from %s with learning curve', env_id, training_dir)
        elif training_video_id is not None:
            logger.info('[%s] Creating evaluation object from %s with training video', env_id, training_dir)
        else:
            raise error.Error("[%s] You didn't have any recorded training data in {}. Once you've used 'env.monitor.start(training_dir)' to start recording, you need to actually run some rollouts. Please join the community chat on https://gym.openai.com if you have any issues.".format(env_id, training_dir))

    evaluation = resource.Evaluation.create(
        training_episode_batch=training_episode_batch_id,
        training_video=training_video_id,
        env=env_info['env_id'],
        algorithm={
            'id': algorithm_id,
        },
        writeup=writeup,
        gym_version=env_info['gym_version'],
        api_key=api_key,
    )

    logger.info(

    """
****************************************************
You successfully uploaded your evaluation on %s to
OpenAI Gym! You can find it at:

    %s

****************************************************
    """.rstrip(), env_id, evaluation.web_url())

    return evaluation

def upload_training_data(training_dir, api_key=None):
    # Could have multiple manifests
    results = monitoring.load_results(training_dir)
    if not results:
        raise error.Error('''Could not find any manifest files in {}.

(HINT: this usually means you did not yet close() your env.monitor and have not yet exited the process. You should call 'env.monitor.start(training_dir)' at the start of training and 'env.monitor.close()' at the end, or exit the process.)'''.format(training_dir))

    manifests = results['manifests']
    env_info = results['env_info']
    timestamps = results['timestamps']
    episode_lengths = results['episode_lengths']
    episode_rewards = results['episode_rewards']
    videos = results['videos']

    env_id = env_info['env_id']
    logger.debug('[%s] Uploading data from manifest %s', env_id, ', '.join(manifests))

    # Do the relevant uploads
    if len(episode_lengths) > 0:
        training_episode_batch = upload_training_episode_batch(episode_lengths, episode_rewards, timestamps, api_key, env_id=env_id)
    else:
        training_episode_batch = None

    if len(videos) > MAX_VIDEOS:
        logger.warn('[%s] You recorded videos for %s episodes, but the scoreboard only supports up to %s. We will automatically subsample for you, but you also might wish to adjust your video recording rate.', env_id, len(videos), MAX_VIDEOS)
        skip = len(videos) / (MAX_VIDEOS - 1)
        videos = videos[::skip]

    if len(videos) > 0:
        training_video = upload_training_video(videos, api_key, env_id=env_id)
    else:
        training_video = None

    return env_info, training_episode_batch, training_video

def upload_training_episode_batch(episode_lengths, episode_rewards, timestamps, api_key=None, env_id=None):
    logger.info('[%s] Uploading %d episodes of training data', env_id, len(episode_lengths))
    file_upload = resource.FileUpload.create(purpose='episode_batch', api_key=api_key)
    file_upload.put({
        'episode_lengths': episode_lengths,
        'episode_rewards': episode_rewards,
        'timestamps': timestamps,
    })
    return file_upload

def upload_training_video(videos, api_key=None, env_id=None):
    """videos: should be list of (video_path, metadata_path) tuples"""
    with tempfile.TemporaryFile() as archive_file:
        write_archive(videos, archive_file, env_id=env_id)
        archive_file.seek(0)

        logger.info('[%s] Uploading videos of %d training episodes (%d bytes)', env_id, len(videos), util.file_size(archive_file))
        file_upload = resource.FileUpload.create(purpose='video', content_type='application/vnd.openai.video+x-compressed', api_key=api_key)
        file_upload.put(archive_file, encode=None)

    return file_upload

def write_archive(videos, archive_file, env_id=None):
    if len(videos) > MAX_VIDEOS:
        raise error.Error('[{}] Trying to upload {} videos, but there is a limit of {} currently. If you actually want to upload this many videos, please email gym@openai.com with your use-case.'.format(env_id, MAX_VIDEOS, len(videos)))

    logger.debug('[%s] Preparing an archive of %d videos: %s', env_id, len(videos), videos)

    # Double check that there are no collisions
    basenames = set()
    manifest = {
        'version': 0,
        'videos': []
    }

    with tarfile.open(fileobj=archive_file, mode='w:gz') as tar:
        for video_path, metadata_path in videos:
            video_name = os.path.basename(video_path)
            metadata_name = os.path.basename(metadata_path)

            if not os.path.exists(video_path):
                raise error.Error('[{}] No such video file {}. (HINT: Your video recorder may have broken midway through the run. You can check this with `video_recorder.functional`.)'.format(env_id, video_path))
            elif not os.path.exists(metadata_path):
                raise error.Error('[{}] No such metadata file {}. (HINT: this should be automatically created when using a VideoRecorder instance.)'.format(env_id, video_path))

            # Do some sanity checking
            if video_name in basenames:
                raise error.Error('[{}] Duplicated video name {} in video list: {}'.format(env_id, video_name, videos))
            elif metadata_name in basenames:
                raise error.Error('[{}] Duplicated metadata file name {} in video list: {}'.format(env_id, metadata_name, videos))
            elif not video_name_re.search(video_name):
                raise error.Error('[{}] Invalid video name {} (must match {})'.format(env_id, video_name, video_name_re.pattern))
            elif not metadata_name_re.search(metadata_name):
                raise error.Error('[{}] Invalid metadata file name {} (must match {})'.format(env_id, metadata_name, metadata_name_re.pattern))

            # Record that we've seen these names; add to manifest
            basenames.add(video_name)
            basenames.add(metadata_name)
            manifest['videos'].append((video_name, metadata_name))

            # Import the files into the archive
            tar.add(video_path, arcname=video_name, recursive=False)
            tar.add(metadata_path, arcname=metadata_name, recursive=False)

        # Actually write the manifest file
        with tempfile.NamedTemporaryFile() as f:
            json.dump(manifest, f)
            f.flush()

            tar.add(f.name, arcname='manifest.json')
