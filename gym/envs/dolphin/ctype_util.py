from ctypes import *
from enum import IntEnum
from itertools import product
import numpy as np
from numpy import random

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
