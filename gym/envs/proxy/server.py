#!/usr/bin/python
"""
Defines a class that listens on an HTTP/websocket port, creates an environement when connected to,
and accepts step & reset calls on that environment.
To use this, you'll want to call
  gym.envs.proxy.server.register('FooEnv-v0', FooEnv)
where FooEnv is a class implementing the Gym Environment protocol.
"""
import math, random, time, logging, re, base64, argparse, collections, sys, os, traceback
import numpy as np
import ujson
from wand.image import Image
import gym
from gym.spaces import Box, Tuple, Discrete
import wsaccel
wsaccel.patch_ws4py()
import cherrypy
from ws4py.server.cherrypyserver import WebSocketPlugin, WebSocketTool
from ws4py.websocket import WebSocket

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def dump_space(s):
    """
    Convert a gym.spaces.Space to something jsonable.
    See .client.load_space
    """
    if isinstance(s, Box):
        return {
            'type': 'Box',
            'low': list(s.low),
            'high': list(s.high),
        }
    elif isinstance(s, Tuple):
        return {
            'type': 'Tuple',
            'spaces': [dump_space(x) for x in s.spaces],
        }
    else:
        raise Exception('Unknown space type %s' % s.__class__.name)

registry = {}
def register(id, cls):
    """
    Register an environement by name. We don't use the regular Gym registry, because there'll already
    be an entry there pointing to GymProxyClient, which would put us in a cycle
    """
    registry[id] = cls

class GymProxyServerSocket(WebSocket):
    def opened(self):
        self.op_count = 0
        logger.info('GymProxyServerSocket opened')

    def received_message(self, message):
        rpc = ujson.loads(message.data)
        logger.debug('rpc > %s', rpc)
        rpc_method = rpc.get('method', None)
        rpc_params = rpc.get('params', None)
        rpc_id = rpc.get('id', None)
        self.op_count += 1
        if self.op_count % 1000 == 0:
            logger.info('%s: %d ops', self.env_name, self.op_count)
        def reply(result, error=None):
            rpc_out = ujson.dumps({
                'id': rpc_id,
                'error': error,
                'result': result,
            })
            self.send(rpc_out)
        try:
            if rpc_method == 'reset':
                reply(self.handle_reset(rpc_params))
            elif rpc_method == 'step':
                reply(self.handle_step(rpc_params))
            elif rpc_method == 'setup':
                reply(self.handle_setup(rpc_params))
            elif rpc_method == 'render':
                reply(self.handle_render(rpc_params))
            elif rpc_method == 'close':
                self.close(reason='requested')
            else:
                raise Exception('unknown method %s' % rpc_method)
        except:
            ex_type, ex_value, ex_tb = sys.exc_info()
            traceback.print_exception(ex_type, ex_value, ex_tb)
            reply(None, ex_type)

    def closed(self, code, reason=None):
        logger.info('GymProxyServer closed %s %s', code, reason)
        pass

    def start_robot(self):
        # override me
        pass

    def handle_reset(self, params):
        obs = self.env.reset()
        return {
            'obs': obs
        }

    def handle_step(self, params):
        action = self.env.action_space.from_jsonable([params['action']])[0]
        obs, reward, done, info = self.env.step(action)
        return {
            'obs': self.env.observation_space.to_jsonable(obs),
            'reward': reward,
            'done': done,
            'info': info,
        }

    def handle_setup(self, params):
        # Override me
        self.env_name = params['env_name']
        logger.info('Creating env %s for client', self.env_name)

        self.env = registry[self.env_name]()

        return {
            'observation_space': dump_space(self.env.observation_space),
            'action_space' : dump_space(self.env.action_space),
            'reward_range': self.env.reward_range,
        }

    def handle_render(self, params):
        mode = params['mode']
        close = params['close']
        img = self.env.render(mode, close)
        return {
            'img': img
        }

def serve_forever(port=9000):

    cherrypy.config.update({'server.socket_port': port})
    WebSocketPlugin(cherrypy.engine).subscribe()
    cherrypy.tools.websocket = WebSocketTool()

    class WebRoot(object):
        @cherrypy.expose
        def index(self):
            return 'some HTML with a websocket javascript connection'

        @cherrypy.expose
        def gymenv(self):
            # you can access the class instance through
            handler = cherrypy.request.ws_handler

    cherrypy.quickstart(WebRoot(), '/', config={'/gymenv': {
        'tools.websocket.on': True,
        'tools.websocket.handler_cls': GymProxyServerSocket
    }})
