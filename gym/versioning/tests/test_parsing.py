import os
import jsonschema
import yaml
from gym.versioning.config_parser import schema

def test_correct_registration():
    with open(os.path.join(os.path.dirname(__file__), 'config.yml')) as f:
        config = yaml.safe_load(f.read())
    try:
        jsonschema.validate(config, yaml.safe_load(schema))
    except jsonschema.exceptions.ValidationError as e:
        assert False, "Caught: {}".format(e)
