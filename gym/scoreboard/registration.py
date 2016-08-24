import collections
import gym.envs
import logging

logger = logging.getLogger(__name__)

class RegistrationError(Exception):
    pass

class Registry(object):
    def __init__(self):
        self.groups = collections.OrderedDict()
        self.envs = collections.OrderedDict()

    def env(self, id):
        return self.envs[id]

    def add_group(self, id, name, description):
        self.groups[id] = {
            'id': id,
            'name': name,
            'description': description,
            'envs': []
        }

    def add_task(self, id, group, summary=None, description=None, background=None, deprecated=False, experimental=False, contributor=None):
        self.envs[id] = {
            'group': group,
            'id': id,
            'summary': summary,
            'description': description,
            'background': background,
            'deprecated': deprecated,
            'experimental': experimental,
            'contributor': contributor,
        }
        if not deprecated:
            self.groups[group]['envs'].append(id)

    def finalize(self, strict=False):
        # Extract all IDs the scoreboard knows about
        scoreboard_ids = set(env_id for group in self.groups.values() for env_id in group['envs'])
        # Extract all IDs gym core knows about, which does not include
        # any external packages
        gym_core_ids = set(spec.id for spec in gym.envs.registry.all() if spec._entry_point and not spec._local_only)

        missing = gym_core_ids - scoreboard_ids
        # Note: it is not an error if the scoreboard contains envs that are
        # not registered in gym core, because some scoreboard envs may come from
        # external packages

        message = []
        if missing:
            message.append('Scoreboard did not register all envs: {}'.format(missing))

        if len(message) > 0:
            message = ' '.join(message)
            if strict:
                raise RegistrationError(message)
            else:
                logger.warn('Site environment registry incorrect: %s', message)

registry = Registry()
add_group = registry.add_group
add_task = registry.add_task
