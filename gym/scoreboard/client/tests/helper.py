import mock
import unittest
import uuid

def fake_id(prefix):
    entropy = ''.join([a for a in str(uuid.uuid4()) if a.isalnum()])
    return '{}_{}'.format(prefix, entropy)

class APITestCase(unittest.TestCase):
    def setUp(self):
        super(APITestCase, self).setUp()
        self.requestor_patcher = mock.patch('gym.scoreboard.client.api_requestor.APIRequestor')
        requestor_class_mock = self.requestor_patcher.start()
        self.requestor_mock = requestor_class_mock.return_value

    def mock_response(self, res):
        self.requestor_mock.request = mock.Mock(return_value=(res, 'reskey'))

class TestData(object):
    @classmethod
    def file_upload_response(cls):
        return {
            'id': fake_id('file'),
            'object': 'file',
        }

    @classmethod
    def evaluation_response(cls):
        return {
            'id': fake_id('file'),
            'object': 'evaluation',
        }
