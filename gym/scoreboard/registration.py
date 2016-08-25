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
        # We used to check whether the scoreboard and environment ID
        # registries matched here. However, we now support various
        # registrations living in various repos, so this is less
        # important.
        pass

registry = Registry()
add_group = registry.add_group
add_task = registry.add_task
