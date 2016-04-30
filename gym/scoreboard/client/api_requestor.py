import json
import platform
import six.moves.urllib as urlparse
from six import iteritems

from gym import error, version
import gym.scoreboard.client
from gym.scoreboard.client import http_client

verify_ssl_certs = True # [SECURITY CRITICAL] only turn this off while debugging
http_client = http_client.RequestsClient(verify_ssl_certs=verify_ssl_certs)

def _build_api_url(url, query):
    scheme, netloc, path, base_query, fragment = urlparse.urlsplit(url)

    if base_query:
        query = '%s&%s' % (base_query, query)

    return urlparse.urlunsplit((scheme, netloc, path, query, fragment))

def _strip_nulls(params):
    if isinstance(params, dict):
        stripped = {}
        for key, value in iteritems(params):
            value = _strip_nulls(value)
            if value is not None:
                stripped[key] = value
        return stripped
    else:
        return params

class APIRequestor(object):
    def __init__(self, key=None, api_base=None):
        self.api_base = api_base or gym.scoreboard.api_base
        self.api_key = key
        self._client = http_client

    def request(self, method, url, params=None, headers=None):
        rbody, rcode, rheaders, my_api_key = self.request_raw(
            method.lower(), url, params, headers)
        resp = self.interpret_response(rbody, rcode, rheaders)
        return resp, my_api_key

    def handle_api_error(self, rbody, rcode, resp, rheaders):
        # Rate limits were previously coded as 400's with code 'rate_limit'
        if rcode == 429:
            raise error.RateLimitError(
                resp.get('detail'), rbody, rcode, resp, rheaders)
        elif rcode in [400, 404]:
            type = resp.get('type')
            if type == 'about:blank':
                type = None
            raise error.InvalidRequestError(
                resp.get('detail'), type,
                rbody, rcode, resp, rheaders)
        elif rcode == 401:
            raise error.AuthenticationError(
                resp.get('detail'), rbody, rcode, resp,
                rheaders)
        else:
            detail = resp.get('detail')

            # This information will only be returned to developers of
            # the OpenAI Gym Scoreboard.
            dev_info = resp.get('dev_info')
            if dev_info:
                detail = "{}\n\n<dev_info>\n{}\n</dev_info>".format(detail, dev_info['traceback'])
            raise error.APIError(detail, rbody, rcode, resp,
                                 rheaders)

    def request_raw(self, method, url, params=None, supplied_headers=None):
        """
        Mechanism for issuing an API call
        """
        if self.api_key:
            my_api_key = self.api_key
        else:
            my_api_key = gym.scoreboard.api_key

        if my_api_key is None:
            raise error.AuthenticationError("""You must provide an OpenAI Gym API key.

(HINT: Set your API key using "gym.scoreboard.api_key = .." or "export OPENAI_GYM_API_KEY=..."). You can find your API key in the OpenAI Gym web interface: https://gym.openai.com/settings/profile.""")

        abs_url = '%s%s' % (self.api_base, url)

        if params:
            encoded_params = json.dumps(_strip_nulls(params))
        else:
            encoded_params = None

        if method == 'get' or method == 'delete':
            if params:
                abs_url = _build_api_url(abs_url, encoded_params)
            post_data = None
        elif method == 'post':
            post_data = encoded_params
        else:
            raise error.APIConnectionError(
                'Unrecognized HTTP method %r.  This may indicate a bug in the '
                'OpenAI Gym bindings.  Please contact gym@openai.com for '
                'assistance.' % (method,))

        ua = {
            'bindings_version': version.VERSION,
            'lang': 'python',
            'publisher': 'openai',
            'httplib': self._client.name,
        }
        for attr, func in [['lang_version', platform.python_version],
                           ['platform', platform.platform]]:
            try:
                val = func()
            except Exception as e:
                val = "!! %s" % (e,)
            ua[attr] = val

        headers = {
            'Openai-Gym-User-Agent': json.dumps(ua),
            'User-Agent': 'Openai-Gym/v1 PythonBindings/%s' % (version.VERSION,),
            'Authorization': 'Bearer %s' % (my_api_key,)
        }

        if method == 'post':
            headers['Content-Type'] = 'application/json'

        if supplied_headers is not None:
            for key, value in supplied_headers.items():
                headers[key] = value

        rbody, rcode, rheaders = self._client.request(
            method, abs_url, headers, post_data)

        return rbody, rcode, rheaders, my_api_key

    def interpret_response(self, rbody, rcode, rheaders):
        content_type = rheaders.get('Content-Type', '')
        if content_type.startswith('text/plain'):
            # Pass through plain text
            resp = rbody

            if not (200 <= rcode < 300):
                self.handle_api_error(rbody, rcode, {}, rheaders)
        else:
            # TODO: Be strict about other Content-Types
            try:
                if hasattr(rbody, 'decode'):
                    rbody = rbody.decode('utf-8')
                resp = json.loads(rbody)
            except Exception:
                raise error.APIError(
                    "Invalid response body from API: %s "
                    "(HTTP response code was %d)" % (rbody, rcode),
                    rbody, rcode, rheaders)

            if not (200 <= rcode < 300):
                self.handle_api_error(rbody, rcode, resp, rheaders)

        return resp
