import sys

# We keep the actual reraising in different modules since the
# Python 2 version SyntaxError's in Python 3.
if sys.version_info[0] < 3:
    from .reraise_impl_py2 import reraise_impl
else:
    from .reraise_impl_py3 import reraise_impl

def reraise(prefix=None, suffix=None):
    old_exc_type, old_exc_value, traceback = sys.exc_info()
    if old_exc_value is None:
        old_exc_value = old_exc_type()
    e = ReraisedException(old_exc_type, old_exc_value, prefix, suffix)

    reraise_impl(e, traceback)

class ReraisedException(Exception):
    def __init__(self, old_exc_type, old_exc_value, prefix, suffix):
        self.old_exc_type = old_exc_type
        self.old_exc_value = old_exc_value
        self.prefix = prefix
        self.suffix = suffix

    def __str__(self):
        klass = self.old_exc_type
        orig = "%s.%s: %s" % (klass.__module__, klass.__name__, klass.__str__(self.old_exc_value))
        prefixpart = suffixpart = ''
        if self.prefix != '':
            prefixpart = self.prefix + "\n"
        if self.suffix != '':
            suffixpart = "\n\n" + self.suffix
        return "%sThe original exception was:\n\n%s%s" % (prefixpart, orig, suffixpart)
