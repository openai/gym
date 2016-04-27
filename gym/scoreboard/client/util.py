import logging
import os
import sys

logger = logging.getLogger(__name__)

def utf8(value):
    if isinstance(value, unicode) and sys.version_info < (3, 0):
        return value.encode('utf-8')
    else:
        return value

def file_size(f):
    return os.fstat(f.fileno()).st_size
