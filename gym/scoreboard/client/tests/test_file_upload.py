from gym.scoreboard.client.tests import helper
from gym import scoreboard

class FileUploadTest(helper.APITestCase):
    def test_create_file_upload(self):
        self.mock_response(helper.TestData.file_upload_response())

        file_upload = scoreboard.FileUpload.create()
        assert isinstance(file_upload, scoreboard.FileUpload), 'File upload is: {!r}'.format(file_upload)

        self.requestor_mock.request.assert_called_with(
            'post',
            '/v1/files',
            params={},
        )
