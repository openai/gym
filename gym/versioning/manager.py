import json
import logging
import requests
import shutil
import os
import gym
import six.moves.urllib as urllib
from gym.envs import register, deregister

from gym.versioning.config_parser import parse_config
from gym.scoreboard.client import http_client, util
from gym.scoreboard.client.resource import UserEnvConfig

logger = logging.getLogger(__name__)

verify_ssl_certs = True # [SECURITY CRITICAL] only turn this off while debugging
http_client = http_client.RequestsClient(verify_ssl_certs=verify_ssl_certs)
gym_abs_path = os.path.dirname(os.path.abspath(gym.__file__))
user_envs_cache_path = 'envs/.user_envs.json'

class VersioningManager(object):
    """
    This object is responsible for downloading and registering user environments (and their versions).
    """
    def __init__(self):
        self.user_envs = []
        self.env_ids = set()
        self.cache_path = os.path.join(gym_abs_path, user_envs_cache_path)

    def load_user_envs(self):
        """ Loads downloaded user envs from filesystem cache on `import gym` """
        if not os.path.exists(self.cache_path):
            return
        with open(self.cache_path) as cache:
            for line in cache:
                self._load_env(line.rstrip('\n'))
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

        # Resetting username and repository if they were not included in config file
        config['username'] = username if 'username' not in config else config['username']
        config['repository'] = repository if 'repository' not in config else config['repository']

        # Parsing (might return more than one env if target_env == '*')
        parsed_envs = parse_config(config, target_env, target_version)
        if len(parsed_envs) == 0:
            return

        # Downloading, and registering
        for parsed_env in parsed_envs:
            env_prefix = parsed_env['id'].replace('-', '_').lower() # username/envname_v0
            env_root_path = os.path.join(gym_abs_path, 'envs', env_prefix)

            if os.path.exists(env_root_path):
                shutil.rmtree(env_root_path)

            for env_file in parsed_env['files']:
                self._download_user_env_file(parsed_env, env_file, env_root_path)

            # Adding missing __init__ files
            if not os.path.isfile(os.path.join(gym_abs_path, 'envs', parsed_env['username'], '__init__.py')):
                open(os.path.join(gym_abs_path, 'envs', parsed_env['username'], '__init__.py'), 'w').close()
            if not os.path.isfile(os.path.join(env_root_path, '__init__.py')):
                open(os.path.join(env_root_path, '__init__.py'), 'w').close()

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
        if os.path.exists(self.cache_path + '.bak'):
            os.remove(self.cache_path + '.bak')
        if os.path.exists(self.cache_path):
            os.rename(self.cache_path, self.cache_path + '.bak')
        with open(self.cache_path, 'w') as cache:
            for user_env in self.user_envs:
                cache.write('{}\n'.format(json.dumps(user_env)))

    def _load_env(self, json_line):
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
                        'file "%s" if this problem persists. \n\nLine: %s', self.cache_path, json_line)
            return None

        self._register(user_env)
        return None

# Have a global manager
manager = VersioningManager()
pull = manager.pull
load_user_envs = manager.load_user_envs
