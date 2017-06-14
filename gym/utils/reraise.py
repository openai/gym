import sys
from six import reraise as reraise_impl, PY3

def reraise(prefix=None, suffix=None):
    old_exc_type, old_exc_value, traceback = sys.exc_info()
    if old_exc_value is None:
        old_exc_value = old_exc_type()

    e = ReraisedException(old_exc_value, prefix, suffix)
    if PY3:
        # Python 3 has exception chaining, which we don't want in this case.
        # Setting `__cause__` to None is equivalent to `from None` syntax
        # which will disable the chaining.
        # See https://www.python.org/dev/peps/pep-0415/
        e.__cause__ = None
    reraise_impl(ReraisedException, e, traceback)

# http://stackoverflow.com/a/13653312
def full_class_name(o):
    module = o.__class__.__module__
    if module is None or module == str.__class__.__module__:
        return o.__class__.__name__
    return module + '.' + o.__class__.__name__

class ReraisedException(Exception):
    def __init__(self, old_exc, prefix, suffix):
        self.old_exc = old_exc
        self.prefix = prefix
        self.suffix = suffix

    def __str__(self):
        klass = self.old_exc.__class__

        orig = "%s: %s" % (full_class_name(self.old_exc), klass.__str__(self.old_exc))
        prefixpart = suffixpart = ''
        if self.prefix is not None:
            prefixpart = self.prefix + "\n"
        if self.suffix is not None:
            suffixpart = "\n\n" + self.suffix
        return "%sThe original exception was:\n\n%s%s" % (prefixpart, orig, suffixpart)
