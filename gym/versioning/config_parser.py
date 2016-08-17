import logging
from jsonschema import validate
from jsonschema.exceptions import ValidationError
import yaml
from importlib import import_module

logger = logging.getLogger(__name__)

expected_params = set([
    'id', 'version', 'entry_point', 'timestep_limit',
    'description', 'trials', 'reward_threshold', 'kwargs',
    'nondeterministic', 'requirements', 'files', 'username',
    'repository', 'commit_hash'])

schema = """
type: object
properties:
    envs:
        type: array
        minItems: 1
        items:
            type: object
            properties:
                id:
                    type: string
                    maxLength: 100
                version:
                    type: integer
                    minimum: 0
                entry_point:
                    type: string
                    minLength: 1
                timestep_limit:
                    type: integer
                    minimum: 1
                    maximum: 999999
                description:
                    type: string
                    maxLength: 999
                trials:
                    type: integer
                    minimum: 1
                    maximum: 9999
                reward_threshold:
                    type: number
                kwargs:
                    type: object
                nondeterministic:
                    type: boolean
                requirements:
                    type: array
                    items:
                        type: string
                files:
                    type: array
                    minItems: 1
                    items:
                        type: string
                username:
                    type: string
                repository:
                    type: string
                commit_hash:
                    type: string

            required:
            - id
            - version
            - requirements
            - files
            - commit_hash

    # Default - If not set per env
    username:
        type: string
    repository:
        type: string
"""


def parse_config(config, target_env, target_version=None):
    try:
        validate(config, yaml.safe_load(schema))
    except ValidationError as err:
        logger.warn('Unable to parse YAML configuration. \nValidation error is: %s', err)
        return []

    found_env = False
    parsed_envs = []
    latest_version = {}
    requirements = set()

    # Finding latest version for each id in config file
    for current_env in config['envs']:
        current_id = current_env['id'].lower()
        if current_id not in latest_version or current_env['version'] > latest_version[current_id]:
            latest_version[current_id] = current_env['version']

    for current_env in config['envs']:
        if current_env['id'].lower() == target_env.lower() or target_env == '*':

            if (target_version is None and current_env['version'] == latest_version[current_env['id'].lower()]) \
                    or current_env['version'] == target_version:

                found_env = True

                user_env = {}
                user_env['username'] = current_env['username'] if 'username' in current_env else config['username']
                user_env['repository'] = current_env['repository'] if 'repository' in current_env else config['repository']
                user_env['env_name'] = current_env['id']
                user_env['id'] = '{}/{}-v{:d}'.format(user_env['username'], current_env['id'], current_env['version'])

                for param in expected_params:
                    if param not in user_env and param in current_env:
                        user_env[param] = current_env[param]

                user_env['entry_point'] = \
                    'gym.envs.{}.{}'.format(
                        user_env['id'].replace('-', '_').replace('/', '.').lower(),
                        user_env['entry_point'])

                extra_params = set(current_env.keys()) - expected_params
                if len(extra_params) > 0:
                    logger.warn('Ignoring extra parameters in the environment configuration file for "%s". '
                                'Parameters: %s', user_env['id'], ', '.join(extra_params))

                for req in user_env['requirements']:
                    requirements.add(req)

                parsed_envs.append(user_env)

    # Checking requirements
    missing_requirements = []
    for req in requirements:
        try:
            import_module(req)
        except ImportError:
            missing_requirements.append(req)

    if len(missing_requirements) > 0:
        logger.warn('The environment(s) depend on the following '
                    'missing requirements: %s', ', '.join(missing_requirements))

    if not found_env:
        logger.warn('No environments were found with the name: "%s"', target_env)

    return parsed_envs
