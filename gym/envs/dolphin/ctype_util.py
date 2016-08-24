from ctypes import *
from enum import IntEnum
from itertools import product
import numpy as np
from numpy import random
import tensorflow as tf

def copy(src, dst):
    """Copies the contents of src to dst"""
    pointer(dst)[0] = src

knownCTypes = set([c_float, c_uint, c_int, c_bool])

def toString(struct):
  fields = [field + "=" + str(getattr(struct, field)) for (field, _) in struct._fields_]
  return "%s{%s}" % (struct.__class__.__name__, ", ".join(fields))

def toTuple(value, ctype=None):
  if ctype is None:
    ctype = type(value)
  if ctype in knownCTypes:
    return value
  if issubclass(ctype, Structure):
    return tuple(toTuple(getattr(value, f), t) for f, t in ctype._fields_)
  # an array type
  return tuple(toTuple(v, ctype._type_) for v in value)

def toDict(value, ctype=None):
  if ctype is None:
    ctype = type(value)
  if ctype in knownCTypes:
    return value
  if issubclass(ctype, Structure):
    return {f: toDict(getattr(value, f), t) for f, t in ctype._fields_}
  # an array type
  return [toDict(v, ctype._type_) for v in value]

def hashStruct(struct):
  return hash(toTuple(struct))

def eqStruct(struct1, struct2):
  return toTuple(struct1) == toTuple(struct2)

def toCType(t):
  if issubclass(t, IntEnum):
    return c_uint
  return t

# class decorator
def pretty_struct(cls):
  cls._fields_ = [(name, toCType(t)) for name, t in cls._fields]
  cls.__repr__ = toString
  cls.__hash__ = hashStruct
  cls.__eq__ = eqStruct
  
  cls.allValues = classmethod(allValues)
  cls.randomValue = classmethod(randomValue)
  
  return cls

def allValues(ctype):
  if issubclass(ctype, IntEnum):
    return list(ctype)
  
  if issubclass(ctype, Structure):
    names, types = zip(*ctype._fields)
    values = [allValues(t) for t in types]
    
    def make(vals):
      obj = ctype()
      for name, val in zip(names, vals):
        setattr(obj, name, val)
      return obj
  
    return [make(vals) for vals in product(*values)]
  
  # TODO: handle bounded ints via _fields
  # TODO: handle arrays
  raise TypeError("Unsupported type %s" % ctype)

def randomValue(ctype):
  if issubclass(ctype, IntEnum):
    return random.choice(list(ctype))
  
  if issubclass(ctype, Structure):
    obj = ctype()
    for name, type_ in ctype._fields:
      setattr(obj, name, randomValue(type_))
    return obj
  
  # TODO: handle arrays
  raise TypeError("Unsupported type %s" % ctype)

# TODO: fill out the rest of this table
ctypes2TF = {
  c_bool : tf.bool,
  c_float : tf.float32,
  c_double : tf.float64,
  c_uint : tf.int64, # no tf.uint32 :(
}

def inputCType(ctype, shape=None, name=""):
  if ctype in ctypes2TF:
    return tf.placeholder(ctypes2TF[ctype], shape, name)
  elif issubclass(ctype, Structure):
    return {f : inputCType(t, shape, name + "/" + f) for (f, t) in ctype._fields_}
  else: # assume an array type
    base_type = ctype._type_
    return [inputCType(base_type, shape, name + "/" + str(i)) for i in range(ctype._length_)]

def constantCTypes(ctype, values, name=""):
  if ctype in ctypes2TF:
    return tf.constant(values, dtype=ctypes2TF[ctype], name=name)
  elif issubclass(ctype, Structure):
    return {f : constantCTypes(t, [getattr(v, f) for v in values], name + "/" + f) for (f, t) in ctype._fields_}
  else: # assume an array type
    base_type = ctype._type_
    return [inputCType(base_type, [v[i] for v in values], name + "/" + str(i)) for i in range(ctype._length_)]

def feedCType(ctype, name, value, feed_dict=None):
  if feed_dict is None:
    feed_dict = {}
  if ctype in ctypes2TF:
    feed_dict[name + ':0'] = value
  elif issubclass(ctype, Structure):
    for f, t in ctype._fields_:
      feedCType(t, name + '/' + f, getattr(value, f), feed_dict)
  else: # assume an array type
    base_type = ctype._type_
    for i in range(ctype._length_):
      feedCType(base_type, name + '/' + str(i), value[i], feed_dict)

  return feed_dict

def feedCTypes(ctype, name, values, feed_dict=None):
  if feed_dict is None:
    feed_dict = {}
  if ctype in ctypes2TF:
    feed_dict[name + ':0'] = values
  elif issubclass(ctype, Structure):
    for f, t in ctype._fields_:
      feedCTypes(t, name + '/' + f, [getattr(v, f) for v in values], feed_dict)
  else: # assume an array type
    base_type = ctype._type_
    for i in range(ctype._length_):
      feedCTypes(base_type, name + '/' + str(i), [v[i] for v in values], feed_dict)

  return feed_dict

def vectorizeCTypes(ctype, values):
  if ctype in ctypes2TF:
    return np.array(values)
  elif issubclass(ctype, Structure):
    return {f : vectorizeCTypes(t, [getattr(v, f) for v in values]) for (f, t) in ctype._fields_}
  else: # assume an array type
    base_type = ctype._type_
    return [vectorizeCTypes(base_type, [v[i] for v in values]) for i in range(ctype._length_)]

