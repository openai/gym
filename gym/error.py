import sys

class Error(Exception):
    pass

# Local errors

class Unregistered(Error):
    """Raised when the user requests an item from the registry that does
    not actually exist.
    """
    pass

class UnregisteredEnv(Unregistered):
    """Raised when the user requests an env from the registry that does
    not actually exist.
    """
    pass

class UnregisteredBenchmark(Unregistered):
    """Raised when the user requests an env from the registry that does
    not actually exist.
    """
    pass

class DeprecatedEnv(Error):
    """Raised when the user requests an env from the registry with an
    older version number than the latest env with the same name.
    """
    pass

class UnseedableEnv(Error):
    """Raised when the user tries to seed an env that does not support
    seeding.
    """
    pass

class DependencyNotInstalled(Error):
    pass

class UnsupportedMode(Exception):
    """Raised when the user requests a rendering mode not supported by the
    environment.
    """
    pass

class ResetNeeded(Exception):
    """When the monitor is active, raised when the user tries to step an
    environment that's already done.
    """
    pass

class ResetNotAllowed(Exception):
    """When the monitor is active, raised when the user tries to step an
    environment that's not yet done.
    """
    pass

class InvalidAction(Exception):
    """Raised when the user performs an action not contained within the
    action space
    """
    pass

# API errors

class APIError(Error):
    def __init__(self, message=None, http_body=None, http_status=None,
                 json_body=None, headers=None):
        super(APIError, self).__init__(message)

        if http_body and hasattr(http_body, 'decode'):
            try:
                http_body = http_body.decode('utf-8')
            except:
                http_body = ('<Could not decode body as utf-8. '
                             'Please report to gym@openai.com>')

        self._message = message
        self.http_body = http_body
        self.http_status = http_status
        self.json_body = json_body
        self.headers = headers or {}
        self.request_id = self.headers.get('request-id', None)

    def __unicode__(self):
        if self.request_id is not None:
            msg = self._message or "<empty message>"
            return u"Request {0}: {1}".format(self.request_id, msg)
        else:
            return self._message

    def __str__(self):
        try:               # Python 2
            return unicode(self).encode('utf-8')
        except NameError:  # Python 3
            return self.__unicode__()


class APIConnectionError(APIError):
    pass


class InvalidRequestError(APIError):

    def __init__(self, message, param, http_body=None,
                 http_status=None, json_body=None, headers=None):
        super(InvalidRequestError, self).__init__(
            message, http_body, http_status, json_body,
            headers)
        self.param = param


class AuthenticationError(APIError):
    pass

class RateLimitError(APIError):
    pass

# Video errors

class VideoRecorderError(Error):
    pass

class InvalidFrame(Error):
    pass

# Wrapper errors

class DoubleWrapperError(Error):
    pass


class WrapAfterConfigureError(Error):
    pass


class RetriesExceededError(Error):
    pass

# Vectorized environments errors

class AlreadySteppingError(Exception):
    """
    Raised when `step` is called asynchronously (e.g. with `step_async`), and
    `step_async` is called again (without a complete call to `step_wait`).
    """
    pass

class AlreadyResettingError(Exception):
    """
    Raised when `reset` is called asynchronously (e.g. with `reset_async`), and
    `reset_async` is called again (without a complete call to `reset_wait`).
    """
    pass

class NotSteppingError(Exception):
    """
    Raised when an asynchronous `step` is not running, but `step_wait` is called.
    """
    pass

class NotResettingError(Exception):
    """
    Raised when an asynchronous `reset` is not running, but `reset_wait` is called.
    """
    pass

class ClosedEnvironmentError(Exception):
    """
    Trying to call `step` or `reset`, while the environment is closed.
    """
    pass
