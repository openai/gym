import requests
import urlparse
import json

class Client(object):
    def __init__(self, remote_base):
        self.remote_base = remote_base

    def _request(self, route, data):
        headers = {'Content-type': 'application/json'}
        resp = requests.post(urlparse.urljoin(self.remote_base, route),
                            data=json.dumps(data),
                            headers=headers)
        resp.raise_for_status()
        return resp
    
    def env_create(self, env_id):
        route = '/v1/envs/'
        data = {'env_id': env_id}
        resp = self._request(route, data)
        instance_id = resp.json()['instance_id']
        return instance_id

    def env_check_exists(self, instance_id):
        route = '/v1/envs/{}/check_exists/'.format(instance_id)
        resp = self._request(route, None)
        exists = resp.json()['exists']
        return exists

    def env_reset(self, instance_id):
        route = '/v1/envs/{}/reset/'.format(instance_id)
        resp = self._request(route, None)
        observation = resp.json()['observation']
        return observation

    def env_step(self, instance_id, action):    
        route = '/v1/envs/{}/step/'.format(instance_id)
        data = {'action': action}
        resp = self._request(route, data)
        observation = resp.json()['observation']
        reward = resp.json()['reward']
        done = resp.json()['done']
        info = resp.json()['info']
        return [observation, reward, done, info]

    def env_monitor_start(self, instance_id, directory,
                              force=False, resume=False):
        route = '/v1/envs/{}/monitor/start/'.format(instance_id)
        data = {'directory': directory,
                'force': force,
                'resume': resume}
        self._request(route, data)

    def env_monitor_close(self, instance_id):
        route = '/v1/envs/{}/monitor/close/'.format(instance_id)
        self._request(route, None)

    def upload(self, training_dir, algorithm_id=None, writeup=None,
                   api_key=None, ignore_open_monitors=False):
        route = '/v1/upload/'
        data = {'training_dir': training_dir,
                'algorithm_id': algorithm_id,
                'writeup': writeup,
                'api_key': api_key,
                'ignore_open_monitors': ignore_open_monitors}
        self._request(route, data)

if __name__ == '__main__':
    remote_base = 'http://127.0.0.1:5000'
    client = Client(remote_base)

    env_id = 'CartPole-v0'
    instance_id = client.env_create(env_id)
    exists = client.env_check_exists(instance_id)
    client.env_monitor_start(instance_id, directory='tmp', force=True)
    init_obs = client.env_reset(instance_id)
    [observation, reward, done, info] = client.env_step(instance_id, 1)
    client.env_monitor_close(instance_id)
    client.upload(training_dir='tmp')
        # NOTE: currently failing because no API_key provided
    


