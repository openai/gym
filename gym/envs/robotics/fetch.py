# Runs within Gym, opens a socket to a container running rosbridge
import numpy as np
from gym import Env, utils
from gym.spaces import Discrete, Tuple
import ujson
import threading
import wsaccel
import logging
wsaccel.patch_ws4py()
from ws4py.client.threadedclient import WebSocketClient

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class GymProxyClient(WebSocketClient):
    def  __init__(self, url):
        self.ws_opened = False
        self.rpc_lock = threading.Lock()
        self.rpc_ready = threading.Condition(self.rpc_lock)
        self.rpc_counter = 485
        self.rpc_pending = []
        WebSocketClient.__init__(self, url, protocols=['http-only', 'chat'])

    def opened(self):
        logger.info('GymProxyClient opened')
        self.rpc_lock = threading.Lock()
        self.rpc_ready = threading.Condition(self.rpc_lock)
        self.rpc_counter = 485
        self.rpc_pending = []

        self.rpc_ready.acquire()
        self.ws_opened = True
        logger.info('self.ws_opened=%s', self.ws_opened)
        self.rpc_ready.notify_all()
        self.rpc_ready.release()

    def closed(self, code, reason=None):
        logger.info('Closed %s %s', code, reason)

    def received_message(self, msg):
        logger.info('received %s', msg)
        rpc_ans = ujson.loads(msg.data)
        self.rpc_ready.acquire()
        self.rpc_pending.append(rpc_ans)
        self.rpc_ready.notify()
        self.rpc_ready.release()

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
        self.rpc_ready.acquire()
        while len(self.rpc_pending) == 0:
            self.rpc_ready.wait(10)
        rpc_ans = self.rpc_pending.pop(0)
        self.rpc_ready.release()
        return rpc_ans

    @classmethod
    def setup(cls, url):
        ret = cls(url, )
        logger.info('Connecting to %s', url)
        ret.connect()
        ws_thread = threading.Thread(target=ret.run_forever)
        return ret


class FetchRobot(Env):
    metadata = {
    }

    def __init__(self):
        self.proxy = GymProxyClient.setup('ws://127.0.0.1:9000/fr')
        self.reset()

    def _step(self, action):
        ret = self.ws.rpc('step', {
            'action': action
        })
        return ret['obs'], ret['reward'], ret['done'], ret['info']

    def _reset(self):
        ret = self.proxy.rpc('reset', {})
        return ret['obs']

    def _render(self, mode='human', close=False):
        # WRITME
        return None
