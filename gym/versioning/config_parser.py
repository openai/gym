import logging
from jsonschema import validate
from jsonschema.exceptions import ValidationError
import yaml
from importlib import import_module
from gym.scoreboard import github_api_key, CommitHash

logger = logging.getLogger(__name__)

expected_params = set([
    'id', 'version', 'entry_point', 'timestep_limit',
    'description', 'trials', 'reward_threshold', 'kwargs',
    'nondeterministic', 'requirements', 'files', 'commit_ref',
    'username', 'repository', 'commit_hash'])

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
                commit_ref:
                    type: string

                # Will be added automatically
                # - username
                # - repository
                # - commit_hash

            required:
            - id
            - version
            - requirements
            - files
            - commit_ref

    # Will be added automatically
    # - username
    # - repository
"""


def parse_config(config, target_env, target_version=None):
    try:
        validate(config, yaml.safe_load(schema))
    except ValidationError as err:
        logger.warn('Unable to parse YAML configuration. \nValidation error is: %s', err)
        return []

    found_env_correct_version = False
    found_env_wrong_version = False
    parsed_envs = []
    latest_version = {}
    requirements = set()
    versions = {}
    commit_hash_cache = {}

    # Finding latest version for each id in config file
    # and making sure multiple versions are not using the same ref
    for current_env in config['envs']:
        current_id = current_env['id'].lower()
        current_repo = config['repository'].lower()
        current_ref = current_env['commit_ref'].lower()
        current_id_repo_ref = '{}|{}|{}'.format(current_id, current_repo, current_ref)
        if current_id not in latest_version or current_env['version'] > latest_version[current_id]:
            latest_version[current_id] = current_env['version']
        if current_id_repo_ref not in versions:
            versions[current_id_repo_ref] = []
        versions[current_id_repo_ref].append(current_env['version'])

    # Printing warning if multiple versions of an environment are using the same commit_ref
    for current_id_repo_ref in versions:
        if len(versions[current_id_repo_ref]) > 1:
            parts = current_id_repo_ref.split('|')
            current_id = parts[0] if len(parts) >= 1 else ''
            current_repo = parts[1] if len(parts) >= 2 else ''
            current_ref = parts[2] if len(parts) >= 3 else ''
            current_versions = ','.join(['"v{}"'.format(x) for x in versions[current_id_repo_ref]])
            logger.warn('Environment "%s" is using the commit ref "%s" of repository "%s" for the following versions: %s. Environments '
                        'are expected to have a different commit ref for each version.', current_id, current_ref, current_repo, current_versions)

    for current_env in config['envs']:
        if current_env['id'].lower() == target_env.lower() or target_env == '*':
            found_env_wrong_version = True

            if (target_version is None and current_env['version'] == latest_version[current_env['id'].lower()]) \
                    or current_env['version'] == target_version:

                found_env_correct_version = True

                user_env = {}
                user_env['username'] = config['username']
                user_env['repository'] = config['repository']
                user_env['env_name'] = current_env['id']
                user_env['id'] = '{}/{}-v{:d}'.format(user_env['username'], user_env['env_name'], current_env['version'])

                for param in expected_params:
                    if param not in user_env and param in current_env:
                        user_env[param] = current_env[param]

                # Retrieving full commit hash from commit ref
                owner_repo_ref = '{}/{}/{}'.format(user_env['username'], user_env['repository'], user_env['commit_ref'])
                if owner_repo_ref in commit_hash_cache:
                    user_env['commit_hash'] = commit_hash_cache[owner_repo_ref]
                else:
                    commit_hash = CommitHash(owner_repo_ref, api_key=github_api_key)
                    commit_hash.refresh()
                    if 'commit_hash' in commit_hash and commit_hash['commit_hash'] is not None:
                        user_env['commit_hash'] = commit_hash['commit_hash']
                        commit_hash_cache[owner_repo_ref] = commit_hash['commit_hash']
                    else:
                        logger.warn('Unable to retrieve 40 character commit hash from commit ref "%s" of repository "%s" owned by '
                                    '"%s". Skipping environment "%s".', user_env['commit_ref'], user_env['repository'],
                                    user_env['username'], user_env['id'])
                        continue

                user_env['entry_point'] = \
                    'gym.envs.custom.{}.{}.{}'.format(
                        user_env['username'].replace('-', '_').replace('/', '.').lower(),
                        user_env['commit_hash'].replace('-', '_').replace('/', '.').lower(),
                        user_env['entry_point'])

                extra_params = set(current_env.keys()) - expected_params
                if len(extra_params) > 0:
                    logger.warn('Ignoring extra parameters in the environment configuration file for "%s". '
                                'Parameters: %s', user_env['id'], ', '.join(extra_params))

                for req in user_env['requirements']:
                    requirements.add(req)

                # Warning for obsolete version
                if current_env['version'] < latest_version[current_env['id'].lower()]:
                    logger.warn(
                        'A more recent version "v%d" of the environment "%s/%s" is available. '
                        'You might want to use that version instead.', latest_version[current_env['id'].lower()],
                        user_env['username'], user_env['env_name'])

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

    if not found_env_correct_version:
        if found_env_wrong_version:
            logger.warn('The environment "%s" does not have a version "%s". The most recent version '
                        'found is "%s".', target_env, target_version, latest_version[target_env.lower()])
        else:
            logger.warn('No environments were found with the name: "%s"', target_env)

    return parsed_envs
