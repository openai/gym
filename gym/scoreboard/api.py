import logging
import json
import os
import re
import tarfile
import tempfile
from gym import benchmark_spec, error, monitoring
from gym.scoreboard.client import resource, util
import numpy as np

MAX_VIDEOS = 100

logger = logging.getLogger(__name__)

video_name_re = re.compile('^[\w.-]+\.(mp4|avi|json)$')
metadata_name_re = re.compile('^[\w.-]+\.meta\.json$')

def upload(training_dir, algorithm_id=None, writeup=None, tags=None, benchmark_id=None, api_key=None, ignore_open_monitors=False):
    """Upload the results of training (as automatically recorded by your
    env's monitor) to OpenAI Gym.

    Args:
        training_dir (Optional[str]): A directory containing the results of a training run.
        algorithm_id (Optional[str]): An algorithm id indicating the particular version of the algorithm (including choices of parameters) you are running (visit https://gym.openai.com/algorithms to create an id). If the id doesn't match an existing server id it will create a new algorithm using algorithm_id as the name
        benchmark_id (Optional[str]): The benchmark that these evaluations belong to. Will recursively search through training_dir for any Gym manifests. This feature is currently pre-release.
        writeup (Optional[str]): A Gist URL (of the form https://gist.github.com/<user>/<id>) containing your writeup for this evaluation.
        tags (Optional[dict]): A dictionary of key/values to store with the benchmark run (ignored for nonbenchmark evaluations). Must be jsonable.
        api_key (Optional[str]): Your OpenAI API key. Can also be provided as an environment variable (OPENAI_GYM_API_KEY).
    """

    if benchmark_id:
        # We're uploading a benchmark run.

        directories = []
        env_ids = []
        for name, _, files in os.walk(training_dir):
            manifests = monitoring.detect_training_manifests(name, files=files)
            if manifests:
                env_info = monitoring.load_env_info_from_manifests(manifests, training_dir)
                env_ids.append(env_info['env_id'])
                directories.append(name)

        # Validate against benchmark spec
        try:
            spec = benchmark_spec(benchmark_id)
        except error.UnregisteredBenchmark:
            raise error.Error("Invalid benchmark id: {}. Are you using a benchmark registered in gym/benchmarks/__init__.py?".format(benchmark_id))

        # TODO: verify that the number of trials matches
        spec_env_ids = [task.env_id for task in spec.tasks for _ in range(task.trials)]

        if not env_ids:
            raise error.Error("Could not find any evaluations in {}".format(training_dir))

        # This could be more stringent about mixing evaluations
        if sorted(env_ids) != sorted(spec_env_ids):
            logger.info("WARNING: Evaluations do not match spec for benchmark %s. In %s, we found evaluations for %s, expected %s", benchmark_id, training_dir, sorted(env_ids), sorted(spec_env_ids))

        benchmark_run = resource.BenchmarkRun.create(benchmark_id=benchmark_id, algorithm_id=algorithm_id, tags=json.dumps(tags))
        benchmark_run_id = benchmark_run.id

        # Actually do the uploads.
        for training_dir in directories:
            # N.B. we don't propagate algorithm_id to Evaluation if we're running as part of a benchmark
            _upload(training_dir, None, writeup, benchmark_run_id, api_key, ignore_open_monitors)

        logger.info("""
****************************************************
You successfully uploaded your benchmark on %s to
OpenAI Gym! You can find it at:

    %s

****************************************************
        """.rstrip(), benchmark_id, benchmark_run.web_url())

        return benchmark_run_id
    else:
        if tags is not None:
             logger.warning("Tags will NOT be uploaded for this submission.")
        # Single evalution upload
        benchmark_run_id = None
        evaluation = _upload(training_dir, algorithm_id, writeup, benchmark_run_id, api_key, ignore_open_monitors)

        logger.info("""
****************************************************
You successfully uploaded your evaluation on %s to
OpenAI Gym! You can find it at:

    %s

****************************************************
        """.rstrip(), evaluation.env, evaluation.web_url())

        return None

def _upload(training_dir, algorithm_id=None, writeup=None, benchmark_run_id=None, api_key=None, ignore_open_monitors=False):
    if not ignore_open_monitors:
        open_monitors = monitoring._open_monitors()
        if len(open_monitors) > 0:
            envs = [m.env.spec.id if m.env.spec else '(unknown)' for m in open_monitors]
            raise error.Error("Still have an open monitor on {}. You must run 'env.close()' before uploading.".format(', '.join(envs)))

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
            raise error.Error("[%s] You didn't have any recorded training data in %s. Once you've used 'env.monitor.start(training_dir)' to start recording, you need to actually run some rollouts. Please join the community chat on https://gym.openai.com if you have any issues."%(env_id, training_dir))

    evaluation = resource.Evaluation.create(
        training_episode_batch=training_episode_batch_id,
        training_video=training_video_id,
        env=env_info['env_id'],
        algorithm={
            'id': algorithm_id,
        },
        benchmark_run_id=benchmark_run_id,
        writeup=writeup,
        gym_version=env_info['gym_version'],
        api_key=api_key,
    )

    return evaluation

def upload_training_data(training_dir, api_key=None):
    # Could have multiple manifests
    results = monitoring.load_results(training_dir)
    if not results:
        raise error.Error('''Could not find any manifest files in {}.

(HINT: this usually means you did not yet close() your env.monitor and have not yet exited the process. You should call 'env.monitor.start(training_dir)' at the start of training and 'env.close()' at the end, or exit the process.)'''.format(training_dir))

    manifests = results['manifests']
    env_info = results['env_info']
    data_sources = results['data_sources']
    timestamps = results['timestamps']
    episode_lengths = results['episode_lengths']
    episode_rewards = results['episode_rewards']
    episode_types = results['episode_types']
    initial_reset_timestamps = results['initial_reset_timestamps']
    videos = results['videos']

    env_id = env_info['env_id']
    logger.debug('[%s] Uploading data from manifest %s', env_id, ', '.join(manifests))

    # Do the relevant uploads
    if len(episode_lengths) > 0:
        training_episode_batch = upload_training_episode_batch(data_sources, episode_lengths, episode_rewards, episode_types, initial_reset_timestamps, timestamps, api_key, env_id=env_id)
    else:
        training_episode_batch = None

    if len(videos) > MAX_VIDEOS:
        logger.warning('[%s] You recorded videos for %s episodes, but the scoreboard only supports up to %s. We will automatically subsample for you, but you also might wish to adjust your video recording rate.', env_id, len(videos), MAX_VIDEOS)
        subsample_inds = np.linspace(0, len(videos)-1, MAX_VIDEOS).astype('int') #pylint: disable=E1101
        videos = [videos[i] for i in subsample_inds]

    if len(videos) > 0:
        training_video = upload_training_video(videos, api_key, env_id=env_id)
    else:
        training_video = None

    return env_info, training_episode_batch, training_video

def upload_training_episode_batch(data_sources, episode_lengths, episode_rewards, episode_types, initial_reset_timestamps, timestamps, api_key=None, env_id=None):
    logger.info('[%s] Uploading %d episodes of training data', env_id, len(episode_lengths))
    file_upload = resource.FileUpload.create(purpose='episode_batch', api_key=api_key)
    file_upload.put({
        'data_sources': data_sources,
        'episode_lengths': episode_lengths,
        'episode_rewards': episode_rewards,
        'episode_types': episode_types,
        'initial_reset_timestamps': initial_reset_timestamps,
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

        f = tempfile.NamedTemporaryFile(mode='w+', delete=False)
        try:
            json.dump(manifest, f)
            f.close()
            tar.add(f.name, arcname='manifest.json')
        finally:
            f.close()
            os.remove(f.name)
