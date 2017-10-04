# Pixels observation for Mujoco on OpenAI Gym

Get observations as pixels (84 by 84, current frame plus the previous three frames stacked together, gray scale) from the Mujoco environments.

The changes made from standard gym are:
1. ```/gym/env/mujoco/mujoco_env_pixel.py``` (new file). Basically save as from ```/gym/env/mujoco/mujoco_env.py```. This is the base class for the new environments we will create that return pixels as observations.
   * Change the class name to ```MujocoEnvPixel```
   * Change ```self.observation_space``` line in ```__init__``` to ```self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 4))```
2. Make a new file for a new env in ```/gym/env/mujoco``` (new file). In the code in this repo I made two envs that return raw pixels: ```half_cheetah_pixel.py``` and ```pusher_pixel.py```. Save as from the existing env and make the following changes
   * change ```from gym.envs.mujoco import mujoco_env``` to ```from gym.envs.mujoco import mujoco_env_pixel```
   * import the modules used for image processing (I use scikit-image)
   ```
   from skimage import color
   from skimage import transform
   ```
   * add "Pixel" to the class name
   * everywhere in the file, change ```mujoco_env.MujocoEnv``` to ```mujoco_env_pixel.MujocoEnvPixel```
   * in ```__init__``` add the line ```self.memory = np.empty([84,84,4],dtype=np.uint8)```
   * change ```_get_obs``` method to be like in this repo (for example, in ```half_cheetah_pixel.py```)
   * don't touch anything else !!!
3. Add imports for the new class and envs in ```/gym/envs/mujoco/__init__.py```
```
from gym.envs.mujoco.mujoco_env_pixel import MujocoEnvPixel
from gym.envs.mujoco.half_cheetah_pixel import HalfCheetahEnvPixel
from gym.envs.mujoco.pusher_pixel import PusherEnvPixel
```
4. Register the new envs in ```/gym/envs/__init__.py``` for example,
```
register(
    id='HalfCheetahPixel-v1',
    entry_point='gym.envs.mujoco:HalfCheetahEnvPixel',
    max_episode_steps=1000,
    reward_threshold=4800.0,
)
```

Test the new env by running the following
```
import gym
import matplotlib.pyplot as plt
env = gym.make('HalfCheetahPixel-v1')
for i_episode in range(2):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
            
plt.imshow(observation[:,:,0], cmap='gray')
plt.show()
```
when the simulation finish you should see observation plotted as an image
