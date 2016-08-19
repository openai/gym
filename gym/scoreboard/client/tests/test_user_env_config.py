from gym.scoreboard.client.tests import helper
from gym import scoreboard

class UserEnvConfigTest(helper.APITestCase):
    def test_retrieve_user_env_config(self):
        self.mock_response(helper.TestData.user_env_config_response())

        user_env_config = scoreboard.UserEnvConfig.retrieve('openai/gym')
        assert isinstance(user_env_config, scoreboard.UserEnvConfig)

        self.requestor_mock.request.assert_called_with(
            'get',
            '/openai/gym/master/.openai.yml',
            {},
            None
        )
