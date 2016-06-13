"""
Helpers to serialize Gym data types over zmq.
It's mostly json, with the following exceptions:
  Numpy arrays are sent in binary
  Numeric inf is sent as {"__type":"number","value":"inf"}
  Spaces (Box, Tuple, Discrete) are handled specially
"""
import math, random, time, logging, base64, sys, traceback
import numpy as np
import ujson
import gym, gym.spaces


def import_blobs(o, parts):
    if isinstance(o, dict):
        t = o.get('__type', None)
        if t is not None:
            if t == 'ndarray':
                return np.frombuffer(parts[o['partno']], dtype=o['dtype']).reshape(o['shape'])
            elif t == 'Box':
                return gym.spaces.Box(low=import_blobs(o['low'], parts), high=import_blobs(o['high'], parts))
            elif t == 'Tuple':
                return gym.spaces.Tuple(spaces=import_blobs(o['spaces'], parts))
            elif t == 'Discrete':
                return gym.spaces.Discrete(n=import_blobs(o['n'], parts))
            elif t == 'tuple':
                return tuple(import_blobs(o['elems'], parts))
            elif t == 'number':
                if o['value'] == 'inf':
                    return np.inf
                elif o['value'] == '-inf':
                    return -np.inf
                else:
                    raise Exception('Unknown value %s' % o['value'])
        else:
            return dict([(k, import_blobs(v, parts)) for k,v in o.items()])
    elif isinstance(o, list):
        return [import_blobs(v, parts) for v in o]
    else:
        return o

def export_blobs(o, parts):
    if isinstance(o, dict):
        return dict([(k, export_blobs(v, parts)) for k,v in o.items()])
    elif isinstance(o, list):
        return [export_blobs(v, parts) for v in o]
    elif isinstance(o, np.ndarray):
        partno = len(parts)
        parts.append(o.tobytes())
        return dict(__type='ndarray', dtype=str(o.dtype), shape=o.shape, partno=partno)
    elif isinstance(o, gym.Space):
        if isinstance(o, gym.spaces.Box):
            return dict(__type='Box', low=export_blobs(o.low, parts), high=export_blobs(o.high, parts))
        elif isinstance(o, gym.spaces.Tuple):
            return dict(__type='Tuple', spaces=export_blobs(o.spaces, parts))
        elif isinstance(o, gym.spaces.Discrete):
            return dict(__type='Discrete', n=export_blobs(o.n, parts))
        else:
            raise Exception('Unknown space %s' % str(o))
    elif o == np.inf:
        return dict(__type='number', value='inf')
    elif o == -np.inf:
        return dict(__type='number', value='-inf')
    elif isinstance(o, tuple):
        return dict(__type='tuple', elems=[export_blobs(v, parts) for v in o])
    return o

def load_msg(rx):
    msg1 = ujson.loads(rx[0])
    msg = import_blobs(msg1, rx)
    return msg

def dump_msg(msg):
    tx = [None]
    msg1 = export_blobs(msg, tx)
    tx[0] = ujson.dumps(msg1)
    return tx

def mk_random_cookie():
    return ''.join([random.choice('123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz') for x in range(12)])
