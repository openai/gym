import os

from gym.scoreboard.client.resource import FileUpload, Evaluation

# Discover API key from the environment. (You should never have to
# change api_base / web_base.)
api_key = os.environ.get('OPENAI_GYM_API_KEY')
api_base = os.environ.get('OPENAI_GYM_API_BASE', 'https://gym-api.openai.com')
web_base = os.environ.get('OPENAI_GYM_WEB_BASE', 'https://gym.openai.com')
