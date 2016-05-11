import sys

# We keep the actual reraising in different modules, since the
# reraising code uses syntax mutually exclusive to Python 2/3.
if sys.version_info[0] < 3:
    from .reraise_impl_py2 import reraise_impl
else:
    from .reraise_impl_py3 import reraise_impl

def reraise(prefix=None, suffix=None):
    old_exc_type, old_exc_value, traceback = sys.exc_info()
    if old_exc_value is None:
        old_exc_value = old_exc_type()

    e = ReraisedException(old_exc_value, prefix, suffix)

    reraise_impl(e, traceback)

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
