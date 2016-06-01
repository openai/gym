from flask import Flask, request, jsonify
import uuid
import gym

class Envs(object):
    def __init__(self):
        self.envs = {}
        self.id_len = 8

    def create(self, env_id):
        env = gym.make(env_id)
        instance_id = str(uuid.uuid4().hex)[:self.id_len]
        self.envs[instance_id] = env
        return instance_id

    def check_exists(self, instance_id):
        return instance_id in self.envs

    def reset(self, instance_id):
        env = self.envs[instance_id]
        obs = env.reset()
        return env.observation_space.to_jsonable(obs)

    def step(self, instance_id, action):
        env = self.envs[instance_id]
        action_from_json = int(env.action_space.from_jsonable(action))
            # TODO: "int" here seems like a red flag that I'm not
            # doing the right thing with JSON. What instead?
        [observation, reward, done, info] = env.step(action_from_json)
        obs_jsonable = env.observation_space.to_jsonable(observation)
        return [obs_jsonable, reward, done, info]

app = Flask(__name__)
envs = Envs()

@app.route('/v1/envs/', methods=['POST'])
def env_create():
    env_id = request.form['env_id']
    instance_id = envs.create(env_id)
    return jsonify(instance_id = instance_id)

@app.route('/v1/envs/<instance_id>/check_exists/', methods=['POST'])
def env_check_exists(instance_id):
    exists = envs.check_exists(instance_id)
    return jsonify(exists = exists)

@app.route('/v1/envs/<instance_id>/reset/', methods=['POST'])
def env_reset(instance_id):    
    observation = envs.reset(instance_id)
    return jsonify(observation = observation)

@app.route('/v1/envs/<instance_id>/step/', methods=['POST'])
def env_step(instance_id):
    action = request.form['action']
    [obs_jsonable, reward, done, info] = envs.step(instance_id, action)
    return jsonify(observation = obs_jsonable,
                    reward = reward, done = done, info = info)

if __name__ == '__main__':
    app.run()
 
# Can test this with
# curl -i -X POST -d env_id='CartPole-v0' http://127.0.0.1:5000/v1/envs/
# curl -i -X POST -d http://127.0.0.1:5000/v1/envs/<instance_id>/check/


