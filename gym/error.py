"""Set of Error classes for gym."""
import warnings


class Error(Exception):
    """Error superclass."""


# Local errors


class Unregistered(Error):
    """Raised when the user requests an item from the registry that does not actually exist."""


class UnregisteredEnv(Unregistered):
    """Raised when the user requests an env from the registry that does not actually exist."""


class NamespaceNotFound(UnregisteredEnv):
    """Raised when the user requests an env from the registry where the namespace doesn't exist."""


class NameNotFound(UnregisteredEnv):
    """Raised when the user requests an env from the registry where the name doesn't exist."""


class VersionNotFound(UnregisteredEnv):
    """Raised when the user requests an env from the registry where the version doesn't exist."""


class UnregisteredBenchmark(Unregistered):
    """Raised when the user requests an env from the registry that does not actually exist."""


class DeprecatedEnv(Error):
    """Raised when the user requests an env from the registry with an older version number than the latest env with the same name."""


class RegistrationError(Error):
    """Raised when the user attempts to register an invalid env. For example, an unversioned env when a versioned env exists."""


class UnseedableEnv(Error):
    """Raised when the user tries to seed an env that does not support seeding."""


class DependencyNotInstalled(Error):
    """Raised when the user has not installed a dependency."""


class UnsupportedMode(Error):
    """Raised when the user requests a rendering mode not supported by the environment."""


class ResetNeeded(Error):
    """When the order enforcing is violated, i.e. step or render is called before reset."""


class ResetNotAllowed(Error):
    """When the monitor is active, raised when the user tries to step an environment that's not yet terminated or truncated."""


class InvalidAction(Error):
    """Raised when the user performs an action not contained within the action space."""


# API errors


class APIError(Error):
    """Deprecated, to be removed at gym 1.0."""

    def __init__(
        self,
        message=None,
        http_body=None,
        http_status=None,
        json_body=None,
        headers=None,
    ):
        """Initialise API error."""
        super().__init__(message)

        warnings.warn("APIError is deprecated and will be removed at gym 1.0")

        if http_body and hasattr(http_body, "decode"):
            try:
                http_body = http_body.decode("utf-8")
            except Exception:
                http_body = "<Could not decode body as utf-8.>"

        self._message = message
        self.http_body = http_body
        self.http_status = http_status
        self.json_body = json_body
        self.headers = headers or {}
        self.request_id = self.headers.get("request-id", None)

    def __unicode__(self):
        """Returns a string, if request_id is not None then make message other use the _message."""
        if self.request_id is not None:
            msg = self._message or "<empty message>"
            return f"Request {self.request_id}: {msg}"
        else:
            return self._message

    def __str__(self):
        """Returns the __unicode__."""
        return self.__unicode__()


class APIConnectionError(APIError):
    """Deprecated, to be removed at gym 1.0."""


class InvalidRequestError(APIError):
    """Deprecated, to be removed at gym 1.0."""

    def __init__(
        self,
        message,
        param,
        http_body=None,
        http_status=None,
        json_body=None,
        headers=None,
    ):
        """Initialises the invalid request error."""
        super().__init__(message, http_body, http_status, json_body, headers)
        self.param = param


class AuthenticationError(APIError):
    """Deprecated, to be removed at gym 1.0."""


class RateLimitError(APIError):
    """Deprecated, to be removed at gym 1.0."""


# Video errors


class VideoRecorderError(Error):
    """Unused error."""


class InvalidFrame(Error):
    """Error message when an invalid frame is captured."""


# Wrapper errors


class DoubleWrapperError(Error):
    """Error message for when using double wrappers."""


class WrapAfterConfigureError(Error):
    """Error message for using wrap after configure."""


class RetriesExceededError(Error):
    """Error message for retries exceeding set number."""


# Vectorized environments errors


class AlreadyPendingCallError(Exception):
    """Raised when `reset`, or `step` is called asynchronously (e.g. with `reset_async`, or `step_async` respectively), and `reset_async`, or `step_async` (respectively) is called again (without a complete call to `reset_wait`, or `step_wait` respectively)."""

    def __init__(self, message: str, name: str):
        """Initialises the exception with name attributes."""
        super().__init__(message)
        self.name = name


class NoAsyncCallError(Exception):
    """Raised when an asynchronous `reset`, or `step` is not running, but `reset_wait`, or `step_wait` (respectively) is called."""

    def __init__(self, message: str, name: str):
        """Initialises the exception with name attributes."""
        super().__init__(message)
        self.name = name


class ClosedEnvironmentError(Exception):
    """Trying to call `reset`, or `step`, while the environment is closed."""


class CustomSpaceError(Exception):
    """The space is a custom gym.Space instance, and is not supported by `AsyncVectorEnv` with `shared_memory=True`."""
