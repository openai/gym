"""
A proxy environment. It looks like a regular environment, but it connects over ZMQ to a
server which runs the actual code.

There's a Python implementation of the other side of this protocol in ./server.py, but most
likely you'd be implementing the server in another language.

"""
import numpy as np
import re, os, threading, logging
from gym import Env, utils, error
try:
    from gym.envs.proxy import zmq_serialize
    import ujson
    import zmq
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install pyzmq and ujson)".format(e))

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class GymProxyClientSocket(object):
    def  __init__(self, url):
        self.url = url
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.REQ)
        logger.info('Connecting to %s...', self.url)
        self.sock.connect(url)

    def rpc(self, method, params):
        tx = zmq_serialize.dump_msg({
            'method': method,
            'params': params,
        })
        if 0: logger.debug('%s < %s', self.url, tx[0])
        self.sock.send_multipart(tx, flags=0, copy=True, track=False)

        rx = self.sock.recv_multipart(flags=0, copy=True, track=False)
        if 0: logger.debug('%s > %s', self.url, rx[0])
        rpc_ans = zmq_serialize.load_msg(rx)

        log_msgs = rpc_ans.get('log_msgs', None)
        if log_msgs is not None and len(log_msgs):
            for log_msg in log_msgs:
                logger.info('%s > %s', self.url, log_msg)

        if rpc_ans['error'] is not None:
            raise Exception('Remote gym server reports: ' + rpc_ans['error'])
        return rpc_ans['result']

    def close(self):
        if self.sock is not None:
            self.sock.close()
            self.sock = None


class GymProxyClient(Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
    }

    def __init__(self, url='tcp://127.0.0.1:6911', env_name=None, **kwargs):
        # Expand environment variable refs in url
        def expand_env(m):
            ret = os.environ.get(m.group(1), None)
            if ret is None:
                logger.warn('No environment var $%s defined', m.group(1))
            return ret
        url = re.sub(r'\$(\w+)', expand_env, url)

        self.proxy = GymProxyClientSocket(url)
        self.session_id = None
        session_owner = 'user %s on %s in %s' % (os.getlogin(), os.uname()[1], os.getcwd())
        setup_args = {
            'env_name': env_name,
            'session_owner': session_owner,
        }
        setup_args.update(kwargs)
        setup_result = self.proxy.rpc('setup', setup_args)
        self.action_space = setup_result['action_space']
        self.observation_space = setup_result['observation_space']
        self.reward_range = tuple(setup_result['reward_range'])
        self.session_id = setup_result['session_id']
        logger.info('GymProxyClient configured action_space=%s observation_space=%s reward_range=%s',
            self.action_space, self.observation_space, self.reward_range)
        self.reset()

    def _step(self, action):
        ret = self.proxy.rpc('step', {
            'action': action,
            'session_id': self.session_id,
        })
        if ret['session_id'] != self.session_id:
            raise Exception('Wrong session id')
        return ret['obs'], ret['reward'], ret['done'], ret['info']

    def _reset(self):
        ret = self.proxy.rpc('reset', {
            'session_id': self.session_id,
        })
        if ret['session_id'] != self.session_id:
            raise Exception('Wrong session id')
        return ret['obs']

    def _render(self, mode='human', close=False):
        ret = self.proxy.rpc('render', {
            'mode': mode,
            'close': close,
            'session_id': self.session_id,
        })
        if ret['session_id'] != self.session_id:
            raise Exception('Wrong session id')
        return ret['img']

    def _close(self):
        if self.session_id is not None:
            self.proxy.rpc('close', {
                'session_id': self.session_id,
            })
            self.session_id = None
        self.proxy.close()
        self.proxy = None
