def getField(obj, field):
    if isinstance(field, str):
        return getattr(obj, field)
    else:
        return obj[field]

def setField(obj, field, val, validate=True):
    if isinstance(field, str):
        if validate:
            if not hasattr(obj, field):
                raise TypeError("%s object has no field %s" % (type(obj), field))
        setattr(obj, field, val)
    else: # assume an array
        obj[field] = val

def getPath(obj, path):
    for field in path:
        obj = getField(obj, field)
    return obj

# doesn't work with empty path :(
def setPath(obj, path, val):
    obj = getPath(obj, path[:-1])
    setField(obj, path[-1], val)

