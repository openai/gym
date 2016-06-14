"""
Helpers to serialize Gym data types over zmq.
"""
import math, logging, random
import numpy as np
import ujson
import gym, gym.spaces

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def mk_random_cookie():
    return ''.join([random.choice('123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz') for x in range(12)])

def dump_msg(msg):
    """
    Turn o into a list of binary strings suitable for sending over zmq.
    This is less general than Pickle, but it's:
        - High-performance
        - Language-neutral. Should be straightforward to read & write from JS or C++
        - Supports Gym objects like subtypes of gym.Space
        - Can be handed untrusted data without creating a remote execution risk

    It returns a list parts, where:
      - parts[0] is a JSON-formatted representation of msg, with some replacements as detailed below.
      - parts[1...] encode bulk data structures like numpy arrays in binary.

    The following replacements are made:
      - numpy arrays are replaced by a description (element type & shape) and reference to the binary
        data stored in parts
      - Numeric inf is sent as {"__type":"number","value":"inf"}
      - Spaces (Box, Tuple, Discrete) are send encoded something like {"__type":"Box","low"=...,"high"=...}
      - tuples and objects are wrapped with a {"__type":"tuple", ...} and {"__type":"object", ...}

    """
    parts = [None]
    msg1 = _dump_msg1(msg, parts)
    parts[0] = ujson.dumps(msg1)
    return parts

def _dump_msg1(o, parts):
    if isinstance(o, dict):
        return dict([(k, _dump_msg1(v, parts)) for k,v in o.items()])
    elif isinstance(o, list):
        return [_dump_msg1(v, parts) for v in o]
    elif isinstance(o, np.ndarray):
        if str(o.dtype) == 'object':
            return dict(__type='ndarray', dtype=str(o.dtype), shape=o.shape, flat=[_dump_msg1(x, parts) for x in o.flat])
        else:
            partno = len(parts)
            parts.append(o.tobytes())
            return dict(__type='ndarray', dtype=str(o.dtype), shape=o.shape, partno=partno)
    elif isinstance(o, gym.Space):
        if isinstance(o, gym.spaces.Box):
            return dict(__type='Box', low=_dump_msg1(o.low, parts), high=_dump_msg1(o.high, parts))
        elif isinstance(o, gym.spaces.Tuple):
            return dict(__type='Tuple', spaces=_dump_msg1(o.spaces, parts))
        elif isinstance(o, gym.spaces.Discrete):
            return dict(__type='Discrete', n=_dump_msg1(o.n, parts))
        else:
            raise Exception('Unknown space %s' % str(o))
    elif isinstance(o, float):
        if np.isposinf(o):
            return dict(__type='number', value='+inf')
        elif np.isneginf(o):
            return dict(__type='number', value='-inf')
        elif np.isnan(o):
            return dict(__type='number', value='nan')
        else:
            return o
    elif isinstance(o, tuple):
        return dict(__type='tuple', elems=[_dump_msg1(v, parts) for v in o])
    else:
        return o

def load_msg(parts):
    """
    Take a list of binary strings received over zmq, and turn it into an object by reversing the transformations
    of dump_msg
    """
    msg1 = ujson.loads(parts[0])
    msg = _load_msg1(msg1, parts)
    return msg

def _load_msg1(o, parts):
    if isinstance(o, dict):
        t = o.get('__type', None)
        if t is not None:
            if t == 'ndarray':
                if o['dtype'] == 'object':
                    return np.array(o['flat']).reshape(o['shape'])
                else:
                    return np.frombuffer(parts[o['partno']], dtype=o['dtype']).reshape(o['shape'])
            elif t == 'Box':
                return gym.spaces.Box(low=_load_msg1(o['low'], parts), high=_load_msg1(o['high'], parts))
            elif t == 'Tuple':
                return gym.spaces.Tuple(spaces=_load_msg1(o['spaces'], parts))
            elif t == 'Discrete':
                return gym.spaces.Discrete(n=_load_msg1(o['n'], parts))
            elif t == 'tuple':
                return tuple(_load_msg1(o['elems'], parts))
            elif t == 'number':
                if o['value'] == '+inf':
                    return np.inf
                elif o['value'] == '-inf':
                    return -np.inf
                elif o['value'] == 'nan':
                    return np.nan
                else:
                    raise Exception('Unknown value %s' % o['value'])
            else:
                logger.warn('Unimplemented object to reconstruct %s', t)
                return o
        else:
            return dict([(k, _load_msg1(v, parts)) for k,v in o.items()])
    elif isinstance(o, list):
        return [_load_msg1(v, parts) for v in o]
    else:
        return o
