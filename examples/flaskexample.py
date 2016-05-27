from flask import Flask, request
import uuid
import gym

class Envs(object):
    def __init__(self):
        self.envs = {}
        self.id_len = 8

    def create(self, env_id):

        # env_id = request.form['env_id']
        # TODO: gdb was able to use the request object directly
        # within the class because the class method was
        # called straight from the blueprint.route

        env = gym.make(env_id)
        instance_id = str(uuid.uuid4().hex)[:self.id_len]
        self.envs[instance_id] = env
        return instance_id

    def check_exists(self, instance_id):
        return instance_id in self.envs

app = Flask(__name__)
envs = Envs()

@app.route('/v1/create/', methods=['POST'])
def create_env():
    instance_id = envs.create(request.form['env_id'])
    # TODO: ok to leave the potential KeyError exposed like this?

    return str(instance_id)
    # TODO: make the returned values be JSON objects

@app.route('/v1/check/', methods=['POST'])
def check_exists():
    return str(envs.check_exists(request.form['instance_id']))

if __name__ == '__main__':
    app.run(debug=True)
 
# Can test this with
# curl -i -X POST -d env_id='CartPole-v0' http://127.0.0.1:5000/v1/create/ 
# curl -i -X POST -d instance_id='<id>' http://127.0.0.1:5000/v1/check/

