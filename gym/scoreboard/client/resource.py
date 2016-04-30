import json
import warnings
import sys
from six import string_types
from six import iteritems
import six.moves.urllib as urllib

import gym
from gym import error
from gym.scoreboard.client import api_requestor, util

def convert_to_gym_object(resp, api_key):
    types = {
        'evaluation': Evaluation,
        'file': FileUpload,
    }

    if isinstance(resp, list):
        return [convert_to_gym_object(i, api_key) for i in resp]
    elif isinstance(resp, dict) and not isinstance(resp, GymObject):
        resp = resp.copy()
        klass_name = resp.get('object')
        if isinstance(klass_name, string_types):
            klass = types.get(klass_name, GymObject)
        else:
            klass = GymObject
        return klass.construct_from(resp, api_key)
    else:
        return resp

def populate_headers(idempotency_key):
    if idempotency_key is not None:
        return {"Idempotency-Key": idempotency_key}
    return None

def _compute_diff(current, previous):
    if isinstance(current, dict):
        previous = previous or {}
        diff = current.copy()
        for key in set(previous.keys()) - set(diff.keys()):
            diff[key] = ""
        return diff
    return current if current is not None else ""

class GymObject(dict):
    def __init__(self, id=None, api_key=None, **params):
        super(GymObject, self).__init__()

        self._unsaved_values = set()
        self._transient_values = set()

        self._retrieve_params = params
        self._previous = None

        object.__setattr__(self, 'api_key', api_key)

        if id:
            self['id'] = id

    def update(self, update_dict):
        for k in update_dict:
            self._unsaved_values.add(k)

        return super(GymObject, self).update(update_dict)

    def __setattr__(self, k, v):
        if k[0] == '_' or k in self.__dict__:
            return super(GymObject, self).__setattr__(k, v)
        else:
            self[k] = v

    def __getattr__(self, k):
        if k[0] == '_':
            raise AttributeError(k)

        try:
            return self[k]
        except KeyError as err:
            raise AttributeError(*err.args)

    def __delattr__(self, k):
        if k[0] == '_' or k in self.__dict__:
            return super(GymObject, self).__delattr__(k)
        else:
            del self[k]

    def __setitem__(self, k, v):
        if v == "":
            raise ValueError(
                "You cannot set %s to an empty string. "
                "We interpret empty strings as None in requests."
                "You may set %s.%s = None to delete the property" % (
                    k, str(self), k))

        super(GymObject, self).__setitem__(k, v)

        # Allows for unpickling in Python 3.x
        if not hasattr(self, '_unsaved_values'):
            self._unsaved_values = set()

        self._unsaved_values.add(k)

    def __getitem__(self, k):
        try:
            return super(GymObject, self).__getitem__(k)
        except KeyError as err:
            if k in self._transient_values:
                raise KeyError(
                    "%r.  HINT: The %r attribute was set in the past."
                    "It was then wiped when refreshing the object with "
                    "the result returned by Rl_Gym's API, probably as a "
                    "result of a save().  The attributes currently "
                    "available on this object are: %s" %
                    (k, k, ', '.join(self.keys())))
            else:
                raise err

    def __delitem__(self, k):
        super(GymObject, self).__delitem__(k)

        # Allows for unpickling in Python 3.x
        if hasattr(self, '_unsaved_values'):
            self._unsaved_values.remove(k)

    @classmethod
    def construct_from(cls, values, key):
        instance = cls(values.get('id'), api_key=key)
        instance.refresh_from(values, api_key=key)
        return instance

    def refresh_from(self, values, api_key=None, partial=False):
        self.api_key = api_key or getattr(values, 'api_key', None)

        # Wipe old state before setting new.  This is useful for e.g.
        # updating a customer, where there is no persistent card
        # parameter.  Mark those values which don't persist as transient
        if partial:
            self._unsaved_values = (self._unsaved_values - set(values))
        else:
            removed = set(self.keys()) - set(values)
            self._transient_values = self._transient_values | removed
            self._unsaved_values = set()
            self.clear()

        self._transient_values = self._transient_values - set(values)

        for k, v in iteritems(values):
            super(GymObject, self).__setitem__(
                k, convert_to_gym_object(v, api_key))

        self._previous = values

    @classmethod
    def api_base(cls):
        return None

    def request(self, method, url, params=None, headers=None):
        if params is None:
            params = self._retrieve_params
        requestor = api_requestor.APIRequestor(
            key=self.api_key, api_base=self.api_base())
        response, api_key = requestor.request(method, url, params, headers)

        return convert_to_gym_object(response, api_key)

    def __repr__(self):
        ident_parts = [type(self).__name__]

        if isinstance(self.get('object'), string_types):
            ident_parts.append(self.get('object'))

        if isinstance(self.get('id'), string_types):
            ident_parts.append('id=%s' % (self.get('id'),))

        unicode_repr = '<%s at %s> JSON: %s' % (
            ' '.join(ident_parts), hex(id(self)), str(self))

        if sys.version_info[0] < 3:
            return unicode_repr.encode('utf-8')
        else:
            return unicode_repr

    def __str__(self):
        return json.dumps(self, sort_keys=True, indent=2)

    def to_dict(self):
        warnings.warn(
            'The `to_dict` method is deprecated and will be removed in '
            'version 2.0 of the Rl_Gym bindings. The GymObject is '
            'itself now a subclass of `dict`.',
            DeprecationWarning)

        return dict(self)

    @property
    def gym_id(self):
        return self.id

    def serialize(self, previous):
        params = {}
        unsaved_keys = self._unsaved_values or set()
        previous = previous or self._previous or {}

        for k, v in self.items():
            if k == 'id' or (isinstance(k, str) and k.startswith('_')):
                continue
            elif isinstance(v, APIResource):
                continue
            elif hasattr(v, 'serialize'):
                params[k] = v.serialize(previous.get(k, None))
            elif k in unsaved_keys:
                params[k] = _compute_diff(v, previous.get(k, None))

        return params

class APIResource(GymObject):
    @classmethod
    def retrieve(cls, id, api_key=None, **params):
        instance = cls(id, api_key, **params)
        instance.refresh()
        return instance

    def refresh(self):
        self.refresh_from(self.request('get', self.instance_path()))
        return self

    @classmethod
    def class_name(cls):
        if cls == APIResource:
            raise NotImplementedError(
                'APIResource is an abstract class.  You should perform '
                'actions on its subclasses (e.g. Charge, Customer)')
        return str(urllib.parse.quote_plus(cls.__name__.lower()))

    @classmethod
    def class_path(cls):
        cls_name = cls.class_name()
        return "/v1/%ss" % (cls_name,)

    def instance_path(self):
        id = self.get('id')
        if not id:
            raise error.InvalidRequestError(
                'Could not determine which URL to request: %s instance '
                'has invalid ID: %r' % (type(self).__name__, id), 'id')
        id = util.utf8(id)
        base = self.class_path()
        extn = urllib.parse.quote_plus(id)
        return "%s/%s" % (base, extn)

class ListObject(GymObject):
    def list(self, **params):
        return self.request('get', self['url'], params)

    def all(self, **params):
        warnings.warn("The `all` method is deprecated and will"
                      "be removed in future versions. Please use the "
                      "`list` method instead",
                      DeprecationWarning)
        return self.list(**params)

    def auto_paging_iter(self):
        page = self
        params = dict(self._retrieve_params)

        while True:
            item_id = None
            for item in page:
                item_id = item.get('id', None)
                yield item

            if not getattr(page, 'has_more', False) or item_id is None:
                return

            params['starting_after'] = item_id
            page = self.list(**params)

    def create(self, idempotency_key=None, **params):
        headers = populate_headers(idempotency_key)
        return self.request('post', self['url'], params, headers)

    def retrieve(self, id, **params):
        base = self.get('url')
        id = util.utf8(id)
        extn = urllib.parse.quote_plus(id)
        url = "%s/%s" % (base, extn)

        return self.request('get', url, params)

    def __iter__(self):
        return getattr(self, 'data', []).__iter__()

# Classes of API operations

class ListableAPIResource(APIResource):
    @classmethod
    def all(cls, *args, **params):
        warnings.warn("The `all` class method is deprecated and will"
                      "be removed in future versions. Please use the "
                      "`list` class method instead",
                      DeprecationWarning)
        return cls.list(*args, **params)

    @classmethod
    def auto_paging_iter(self, *args, **params):
        return self.list(*args, **params).auto_paging_iter()

    @classmethod
    def list(cls, api_key=None, idempotency_key=None, **params):
        requestor = api_requestor.APIRequestor(api_key)
        url = cls.class_path()
        response, api_key = requestor.request('get', url, params)
        return convert_to_gym_object(response, api_key)


class CreateableAPIResource(APIResource):
    @classmethod
    def create(cls, api_key=None, idempotency_key=None, **params):
        requestor = api_requestor.APIRequestor(api_key)
        url = cls.class_path()
        headers = populate_headers(idempotency_key)
        response, api_key = requestor.request('post', url, params, headers)
        return convert_to_gym_object(response, api_key)


class UpdateableAPIResource(APIResource):
    def save(self, idempotency_key=None):
        updated_params = self.serialize(None)
        headers = populate_headers(idempotency_key)

        if updated_params:
            self.refresh_from(self.request('post', self.instance_path(),
                                           updated_params, headers))
        else:
            util.logger.debug("Trying to save already saved object %r", self)
        return self


class DeletableAPIResource(APIResource):
    def delete(self, **params):
        self.refresh_from(self.request('delete', self.instance_path(), params))
        return self

## Our resources

class FileUpload(ListableAPIResource):
    @classmethod
    def class_name(cls):
        return 'file'

    @classmethod
    def create(cls, api_key=None, **params):
        requestor = api_requestor.APIRequestor(
            api_key, api_base=cls.api_base())
        url = cls.class_path()
        response, api_key = requestor.request(
            'post', url, params=params)
        return convert_to_gym_object(response, api_key)

    def put(self, contents, encode='json'):
        supplied_headers = {
            "Content-Type": self.content_type
        }
        if encode == 'json':
            contents = json.dumps(contents)
        elif encode is None:
            pass
        else:
            raise error.Error('Encode request for put must be "json" or None, not {}'.format(encode))

        files = {'file': contents}

        body, code, headers = api_requestor.http_client.request(
            'post', self.post_url, post_data=self.post_fields, files=files, headers={})
        if code != 204:
            raise error.Error("Upload to S3 failed. If error persists, please contact us at gym@openai.com this message. S3 returned '{} -- {}'. Tried 'POST {}' with fields {}.".format(code, body, self.post_url, self.post_fields))

class Evaluation(CreateableAPIResource):
    def web_url(self):
        return "%s/evaluations/%s" % (gym.scoreboard.web_base, self.get('id'))
