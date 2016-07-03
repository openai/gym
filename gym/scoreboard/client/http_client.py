import logging
import requests
import textwrap
import six

from gym import error
from gym.scoreboard.client import util

logger = logging.getLogger(__name__)
warned = False

def render_post_data(post_data):
    if hasattr(post_data, 'fileno'): # todo: is this the right way of checking if it's a file?
        return '%r (%d bytes)' % (post_data, util.file_size(post_data))
    elif isinstance(post_data, (six.string_types, six.binary_type)):
        return '%r (%d bytes)' % (post_data, len(post_data))
    else:
        return None

class RequestsClient(object):
    name = 'requests'

    def __init__(self, verify_ssl_certs=True):
        self._verify_ssl_certs = verify_ssl_certs
        self.session = requests.Session()

    def request(self, method, url, headers, post_data=None, files=None):
        global warned
        kwargs = {}

        # Really, really only turn this off while debugging.
        if not self._verify_ssl_certs:
            if not warned:
                logger.warn('You have disabled SSL cert verification in OpenAI Gym, so we will not verify SSL certs. This means an attacker with control of your network could snoop on or modify your data in transit.')
                warned = True
            kwargs['verify'] = False

        try:
            try:
                result = self.session.request(method,
                                              url,
                                              headers=headers,
                                              data=post_data,
                                              timeout=200,
                                              files=files,
                                              **kwargs)
            except TypeError as e:
                raise TypeError(
                    'Warning: It looks like your installed version of the '
                    '"requests" library is not compatible with OpenAI Gym\'s'
                    'usage thereof. (HINT: The most likely cause is that '
                    'your "requests" library is out of date. You can fix '
                    'that by running "pip install -U requests".) The '
                    'underlying error was: %s' % (e,))

            # This causes the content to actually be read, which could cause
            # e.g. a socket timeout. TODO: The other fetch methods probably
            # are susceptible to the same and should be updated.
            content = result.content
            status_code = result.status_code
        except Exception as e:
            # Would catch just requests.exceptions.RequestException, but can
            # also raise ValueError, RuntimeError, etc.
            self._handle_request_error(e, method, url)

        if logger.level <= logging.DEBUG:
            logger.debug(
            """API request to %s returned (response code, response body) of
(%d, %r)

Request body was: %s""", url, status_code, content, render_post_data(post_data))
        elif logger.level <= logging.INFO:
            logger.info('HTTP request: %s %s %d', method.upper(), url, status_code)
        return content, status_code, result.headers

    def _handle_request_error(self, e, method, url):
        if isinstance(e, requests.exceptions.RequestException):
            msg = ("Unexpected error communicating with OpenAI Gym "
                   "(while calling {} {}). "
                   "If this problem persists, let us know at "
                   "gym@openai.com.".format(method, url))
            err = "%s: %s" % (type(e).__name__, str(e))
        else:
            msg = ("Unexpected error communicating with OpenAI Gym. "
                   "It looks like there's probably a configuration "
                   "issue locally.  If this problem persists, let us "
                   "know at gym@openai.com.")
            err = "A %s was raised" % (type(e).__name__,)
            if str(e):
                err += " with error message %s" % (str(e),)
            else:
                err += " with no error message"
        msg = textwrap.fill(msg, width=140) + "\n\n(Network error: %s)" % (err,)
        raise error.APIConnectionError(msg)
