import json
import logging
import requests
import os
import gym
import six.moves.urllib as urllib
from gym.envs import register, deregister

from gym.versioning.config_parser import parse_config
from gym.scoreboard.client import http_client, util
from gym.scoreboard import UserEnvConfig

logger = logging.getLogger(__name__)

verify_ssl_certs = True # [SECURITY CRITICAL] only turn this off while debugging
http_client = http_client.RequestsClient(verify_ssl_certs=verify_ssl_certs)
gym_abs_path = os.path.dirname(os.path.abspath(gym.__file__))
user_env_cache_name = '.envs.json'

class VersioningManager(object):
    """
    This object is responsible for downloading and registering user environments (and their versions).
    """
    def __init__(self):
        self.user_envs = []
        self.env_ids = set()
        self.custom_env_path = os.path.join(gym_abs_path, 'envs', 'custom')

    def load_user_envs(self):
        """ Loads downloaded user envs from filesystem cache on `import gym` """
        if not os.path.isdir(self.custom_env_path):
            return
        for directory in os.listdir(self.custom_env_path):
            current_cache_path = os.path.join(self.custom_env_path, directory, user_env_cache_name)
            if os.path.isfile(current_cache_path):
                with open(current_cache_path) as cache:
                    for line in cache:
                        self._load_env(current_cache_path, line.rstrip('\n'))
        if len(self.env_ids) > 0:
            logger.info('Found and registered %d user environments.', len(self.env_ids))

    def pull(self, env_name='', version=None):
        """
        Downloads and registers a user environment from a Github repository
        Args:
            env_name: the name of the environment (format: username/repository/env-name)
            version: (optional, int) the version number of the environment to download (or latest version will be downloaded by default)

        Note: the user environment will be registered as (username/EnvName-vVersion)
        """
        # Checking syntax
        user_env_parts = env_name.split('/')
        if len(user_env_parts) != 3:
            logger.warn(""" Invalid Syntax - env-name must be in the format username/repository/env-name

Syntax: gym.pull('username/repository/[envname|*]'[, version=integer])

where username is a GitHub username, repository is the name of a GitHub repository, and env-name is the environment name.
The repository must have a ".openai.yml" in its top-level folder on its master branch.""")
            return

        if version is None:
            target_version = version
        else:
            try:
                target_version = int(version)
            except ValueError:
                logger.warn(""" Invalid Syntax - version must be an integer

Syntax: gym.pull('username/repository/[envname|*]'[, version=integer])

where username is a GitHub username, repository is the name of a GitHub repository, and env-name is the environment name.
The repository must have a ".openai.yml" in its top-level folder on its master branch.""")
                return

        username = user_env_parts[0]
        repository = user_env_parts[1]
        target_env = user_env_parts[2]
        try:
            config = UserEnvConfig(id='{}/{}'.format(username, repository))  # id is username/repo
            config.refresh()
        except gym.error.APIError as err:
            if err.http_status == 404:
                logger.warn('Error 404 - Configuration file not found. URL: "%s%s".', UserEnvConfig.api_base(), config.instance_path())
                return
            else:
                raise err

        # Setting username and repository on config top-level
        config['username'] = username
        config['repository'] = repository

        # Parsing (might return more than one env if target_env == '*')
        parsed_envs = parse_config(config, target_env, target_version)
        if len(parsed_envs) == 0:
            return

        # Downloading, and registering
        for parsed_env in parsed_envs:
            env_prefix = '{}/{}'.format(parsed_env['username'], parsed_env['commit_hash']).replace('-', '_').lower() # username/commit_hash
            env_root_path = os.path.join(self.custom_env_path, env_prefix)

            for env_file in parsed_env['files']:
                self._download_user_env_file(parsed_env, env_file, env_root_path)

            # Adding missing __init__ files
            target_files = [
                os.path.join(self.custom_env_path, '__init__.py'),
                os.path.join(self.custom_env_path, parsed_env['username'], '__init__.py'),
                os.path.join(self.custom_env_path, parsed_env['username'], parsed_env['commit_hash'], '__init__.py')
            ]
            for target_file in target_files:
                current_target = target_file.replace('-', '_').lower()
                if not os.path.isfile(current_target):
                    open(current_target, 'w').close()

            if self._register(parsed_env):
                logger.info('Registered the environment: "%s"', parsed_env['id'])

        self._update_cache()
        return

    def _download_user_env_file(self, user_env, env_file, env_root_path):
        target_url = '{}/{}/{}/{}/{}'.format(
            gym.scoreboard.github_raw_base,
            urllib.parse.quote_plus(util.utf8(user_env['username'])),
            urllib.parse.quote_plus(util.utf8(user_env['repository'])),
            urllib.parse.quote_plus(util.utf8(user_env['commit_hash'])),
            urllib.parse.quote_plus(util.utf8(env_file)).replace('%2F', '/'))
        target_file = os.path.join(env_root_path, env_file)

        if os.path.isfile(target_file):
            return  # Already cached for this commit hash, no need to re-download
        if not os.path.exists(os.path.dirname(target_file)):
            os.makedirs(os.path.dirname(target_file))
        with open(target_file, 'wb') as handle:
            response = requests.get(target_url, stream=True, timeout=60)

            if logger.level <= logging.DEBUG:
                logger.debug(
                    'API request to %s returned (response code) of\n(%d)',
                    target_url, response.status_code)

            if not (200 <= response.status_code < 300):
                logger.warn('Unable to download file "%s" for env "%s". The env might not work properly and '
                            'you might need to re-pull it.\nRequest URL: %s\nResponse Code:%d',
                            env_file, user_env['id'], target_file, response.status_code)
                # Removing possibly corrupt file from cache
                if os.path.isfile(target_file):
                    os.remove(target_file)
                return

            for block in response.iter_content(1024):
                handle.write(block)

    def _register(self, user_env):
        if user_env['id'].lower() in self.env_ids:
            logger.info('Deregistering the environment %s, because it is already registered.', user_env['id'])
            deregister(user_env['id'])
            for i, env in enumerate(self.user_envs):
                if env['id'].lower() == user_env['id'].lower():
                    self.user_envs.pop(i)
                    break

        register_params = [
            'id', 'entry_point', 'timestep_limit', 'trials', 'reward_threshold', 'local_only', 'kwargs',
            'nondeterministic', 'wrappers']
        register_kwargs = { k: user_env[k] for k in register_params if k in user_env }
        register(**register_kwargs)
        self.user_envs.append(user_env)
        self.env_ids.add(user_env['id'].lower())
        return True

    def _update_cache(self):
        envs_by_username = {}
        for user_env in self.user_envs:
            username = user_env['username'].replace('-', '_').lower()
            if username not in envs_by_username:
                envs_by_username[username] = []
            envs_by_username[username].append(user_env)

        for username in envs_by_username:
            cache_file = os.path.join(self.custom_env_path, username, user_env_cache_name)
            if not os.path.exists(os.path.dirname(cache_file)):
                os.makedirs(os.path.dirname(cache_file))
            with open(cache_file, 'w') as cache:
                for user_env in envs_by_username[username]:
                    cache.write('{}\n'.format(json.dumps(user_env)))

    def _load_env(self, current_cache_path, json_line):
        if len(json_line) == 0:
            return

        valid_json = False
        try:
            user_env = json.loads(json_line)
            valid_json = True
        except ValueError:
            pass

        if not valid_json or not 'id' in user_env:
            logger.warn('Unable to load user environment. Try deleting your cache '
                        'file "%s" if this problem persists. \n\nLine: %s', current_cache_path, json_line)
            return None

        self._register(user_env)
        return None

# Have a global manager
manager = VersioningManager()
pull = manager.pull
load_user_envs = manager.load_user_envs
