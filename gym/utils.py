"""A set of common utilities used within the environments. These are
not intended as API functions, and will not remain stable over time.
"""

import six

color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)


def colorize(string, color, bold=False, highlight = False):
    """Return string surrounded by appropriate terminal color codes to
    print colorized text.  Valid colors: gray, red, green, yellow,
    blue, magenta, cyan, white, crimson
    """
    attr = []
    num = color2num[color]
    if highlight: num += 10
    attr.append(six.u(str(num)))
    if bold: attr.append(six.u('1'))
    attrs = six.u(';').join(attr)
    return six.u('\x1b[%sm%s\x1b[0m') % (attrs, string)

class EzPickle(object):
    """Objects that are pickled and unpickled via their constructor
    arguments.

    Example usage:

        class Dog(Animal, EzPickle):
            def __init__(self, furcolor, tailkind="bushy"):
                Animal.__init__()
                EzPickle.__init__(furcolor, tailkind)
                ...

    When this object is unpickled, a new Dog will be constructed by passing the provided
    furcolor and tailkind into the constructor. However, philosophers are still not sure
    whether it is still the same dog.

    This is generally needed only for environments which wrap C/C++ code, such as MuJoCo
    and Atari.
    """
    def __init__(self, *args, **kwargs):
        self._ezpickle_args = args
        self._ezpickle_kwargs = kwargs
    def __getstate__(self):
        return {"_ezpickle_args" : self._ezpickle_args, "_ezpickle_kwargs": self._ezpickle_kwargs}
    def __setstate__(self, d):
        out = type(self)(*d["_ezpickle_args"], **d["_ezpickle_kwargs"])
        self.__dict__.update(out.__dict__)
