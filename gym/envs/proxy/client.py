"""
A proxy environment. It looks like a regular environment, but it connects over a websocket
to a server which runs the actual code.

There's a Python implementation of the other side of this protocol in ./server.py, but most likely you'd be
implementing the server in another language.

"""
import numpy as np
import re, os
from gym import Env, utils, error
from gym.spaces import Box, Discrete, Tuple
import threading
import logging
try:
    import ujson
    import wsaccel
    wsaccel.patch_ws4py()
    from ws4py.client.threadedclient import WebSocketClient
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install ws4py, wsaccel and ujson)".format(e))

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def load_space(s):
    if s['type'] == 'Box':
        return Box(np.array(s['low']), np.array(s['high']))
    elif s['type'] == 'Tuple':
        return Tuple([load_space(x) for x in s['spaces']])
    else:
        raise Exception('Unknown space type %s' % s)

class GymProxyClientSocket(WebSocketClient):
    def  __init__(self, url):
        self.ws_opened = False
        self.rpc_lock = threading.Lock()
        self.rpc_ready = threading.Condition(self.rpc_lock)
        self.rpc_counter = 485
        self.rpc_pending = []
        WebSocketClient.__init__(self, url, protocols=['http-only', 'chat'])

    def opened(self):
        logger.info('GymProxyClient opened')
        with self.rpc_ready:
            self.ws_opened = True
            self.rpc_ready.notify_all()

    def closed(self, code, reason=None):
        logger.info('Closed %s %s', code, reason)

    def received_message(self, msg):
        logger.debug('received %s', msg)
        rpc_ans = ujson.loads(msg.data)
        with self.rpc_ready:
            self.rpc_pending.append(rpc_ans)
            self.rpc_ready.notify()

    def rpc(self, method, params):
        with self.rpc_ready:
            while not self.ws_opened:
                logger.info('self.ws_opened==%s (waiting)', self.ws_opened)
                self.rpc_ready.wait(5)

        rpc_id = self.rpc_counter
        self.rpc_counter += 1
        self.send(ujson.dumps({
            'method': method,
            'params': params,
            'id': rpc_id,
        }))

        # This RPC mechanism only allows a single outstanding query at a time
        with self.rpc_ready:
            while len(self.rpc_pending) == 0:
                self.rpc_ready.wait(10)
            rpc_ans = self.rpc_pending.pop(0)
        assert rpc_ans['id'] == rpc_id

        if rpc_ans['error'] is not None:
            raise Exception(rpc_ans['error'])
        return rpc_ans['result']

    @classmethod
    def setup(cls, url):
        ret = cls(url)
        logger.info('Connecting to %s', url)
        ret.connect()
        ws_thread = threading.Thread(target=ret.run_forever)
        ws_thread.daemon = True
        ws_thread.start()
        return ret


class GymProxyClient(Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
    }

    def __init__(self, url='ws://127.0.0.1:9000/gymenv', **kwargs):

        # Expand environment variable refs in url
        def expand_env(m):
            ret = os.environ.get(m.group(1), None)
            if ret is None:
                logger.warn('No environment var $%s defined', m.group(1))
            return ret
        url = re.sub(r'\$(\w+)', expand_env, url)

        self.proxy = GymProxyClientSocket.setup(url)
        setup_result = self.proxy.rpc('setup', kwargs)
        self.action_space = load_space(setup_result['action_space'])
        self.observation_space = load_space(setup_result['observation_space'])
        self.reward_range = tuple(setup_result['reward_range'])
        self.reset()

    def _step(self, action):
        ret = self.proxy.rpc('step', {
            'action': action,
        })
        return self.observation_space.from_jsonable([ret['obs']])[0], ret['reward'], ret['done'], ret['info']

    def _reset(self):
        ret = self.proxy.rpc('reset', {})
        return self.observation_space.from_jsonable([ret['obs']])[0]

    def _render(self, mode='human', close=False):
        ret = self.proxy.rpc('render', {
            'mode': mode,
            'close': close,
        })
        return np.array(ret['img'], dtype=np.uint8)
