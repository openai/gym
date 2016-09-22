import gym
from gym import spaces

import numpy as np
import random

import os
import time
from warnings import warn

from . import ssbm, state_manager, memory_watcher, movie, util
from .menu_manager import *
from .state import *
from .pad import Pad
from .dolphin import runDolphin
from . import ctype_util as ctutil
from .reward import computeRewards

buttons = ['A', 'B', 'Y', 'L', 'Z']

button_space = spaces.Discrete(len(buttons) + 1)
main_stick_space = spaces.Box(0, 1, [2]) # X and Y axes

c_directions = [(0.5, 0.5), (0.5, 1), (0.5, 0), (0, 0.5), (1, 0.5)]
c_stick_space = spaces.Discrete(len(c_directions))

controller_space = spaces.Tuple((button_space, main_stick_space, c_stick_space))

def realController(control):
  button, main, c = control
  
  controller = ssbm.RealControllerState()

  if button < len(buttons):
    setattr(controller, 'button_' + buttons[button], True)
  
  controller.stick_MAIN = tuple(main)
  controller.stick_C = c_directions[c]
  
  return controller

class BoolConv:
  def __init__(self):
    self.space = spaces.Discrete(2)
  def __call__(self, cbool, name=None):
    #assert(type(cbool) is bool, "")
    return int(cbool)

boolConv = BoolConv()

def clip(x, min_x, max_x):
  return min(max(x, min_x), max_x)

class RealConv:
  def __init__(self, low, high, verbose=True):
    self.low = low
    self.high = high
    self.space = spaces.Box(low, high, [1])
    self.verbose = verbose
  
  def __call__(self, x, name=None):
    if self.low > x or x > self.high:
      if self.verbose:
        warn("%f out of bounds in real space \"%s\"" % (x, name))
      x = clip(x, self.low, self.high)
    return np.array([x])

class DiscreteConv:
  def __init__(self, size):
    self.space = spaces.Discrete(size)
  
  def __call__(self, x, name=None, verbose=True):
    if 0 > x or x >= self.space.n:
      if verbose:
        warn("%d out of bounds in discrete space \"%s\"" % (x, name))
      x = 0
    return x

class StructConv:
  def __init__(self, spec):
    self.spec = spec
    
    self.space = spaces.Tuple([conv.space for _, conv in spec])
  
  def __call__(self, struct, **kwargs):
    return [conv(getattr(struct, name), name=name) for name, conv in self.spec]

class ArrayConv:
  def __init__(self, conv, permutation):
    self.conv = conv
    self.permutation = permutation
    
    self.space = spaces.Tuple([conv.space for _ in permutation])
  
  def __call__(self, array, **kwargs):
    return [self.conv(array[i]) for i in self.permutation]

maxCharacter = 32 # should be large enough?

maxAction = 0x017E
numActions = 1 + maxAction

frameConv = RealConv(0, 100, 'frame')
speedConv = RealConv(-20, 20, 'speed') # generally less than 1 in magnitude

player_spec = [
  ('percent', RealConv(0, 200)),
  ('facing', RealConv(-1, 1)),
  ('x', RealConv(-200, 200)),
  ('y', RealConv(-200, 200)),
  ('action_state', DiscreteConv(numActions)),
  ('action_frame', frameConv),
  ('character', DiscreteConv(maxCharacter)),
  ('invulnerable', boolConv),
  ('hitlag_frames_left', frameConv),
  ('hitstun_frames_left', frameConv),
  ('jumps_used', DiscreteConv(8)),
  ('charging_smash', boolConv),
  ('in_air', boolConv),
  ('speed_air_x_self', speedConv),
  ('speed_ground_x_self', speedConv),
  ('speed_y_self', speedConv),
  ('speed_x_attack', speedConv),
  ('speed_y_attack', speedConv),
  ('shield_size', RealConv(0, 100)),
]

playerConv = StructConv(player_spec)

def gameSpec(self=0, enemy=1, swap=False):
  players = [self, enemy]
  if swap:
    players.reverse()
  
  return [
    ('players', ArrayConv(playerConv, players)),
    ('stage', DiscreteConv(32)),
  ]

game_spec = gameSpec()
gameConv = StructConv(game_spec)

gameConv1 = StructConv(gameSpec(swap=True))

default_args = dict(
  tag=None,
  dolphin_dir = 'dolphin/',
  self_play = True,
  zmq=True,
  p1="marth",
  p2="fox",
  stage="battlefield",
  iso="SSBM.iso",
)

class SmashEnv(gym.Env):
  def __init__(self, **kwargs):
    for k, v in default_args.items():
      if k in kwargs and kwargs[k] is not None:
          setattr(self, k, kwargs[k])
      else:
          setattr(self, k, v)
                
    self.observation_space = gameConv.space
    self.action_space = controller_space
    self.realController = realController
    
    self.first_frame = True
    self.toggle = False

    if self.tag is None:
      self.tag = random.getrandbits(32)
    
    self.dolphin_dir += str(self.tag) + "/"
    #self.dolphin_dir = os.path.expanduser(self.dolphin_dir)

    self.prev_state = ssbm.GameMemory()
    self.state = ssbm.GameMemory()
    # track players 1 and 2 (pids 0 and 1)
    self.sm = state_manager.StateManager([0, 1])
    self.write_locations()
    
    self.cpus = [0, 1]# if self.self_play else [0]
    self.characters = [self.p1, self.p2] if self.self_play else [self.p1]

    # sets the game mode and random stage
    self.movie = movie.Movie(movie.endless_netplay_battlefield)

    print('Creating MemoryWatcher.')
    mwType = memory_watcher.MemoryWatcher
    if self.zmq:
      mwType = memory_watcher.MemoryWatcherZMQ
    self.mw = mwType(self.dolphin_dir + '/MemoryWatcher/MemoryWatcher')
    
    pipe_dir = self.dolphin_dir + '/Pipes/'
    print('Creating Pads at %s.' % pipe_dir)
    os.makedirs(self.dolphin_dir + '/Pipes/', exist_ok=True)
    
    paths = [pipe_dir + 'phillip%d' % i for i in self.cpus]
    self.get_pads = util.async_map(Pad, paths)
    
    time.sleep(1) # give pads time to set up
    
    config = dict(
      user=self.dolphin_dir,
      cpus=self.cpus,
      cpu_thread=True,
      gfx='OGL',
      #exe='dolphin-emu-headless',
    )
    config.update(kwargs)
    
    self.dolphin_process = runDolphin(**config)
    
    try:
      self.pads = self.get_pads()
    except KeyboardInterrupt:
      print("Pipes not initialized!")
      return
    
    self.menu_managers = [MenuManager(characters[c], pid=i, pad=p) for c, i, p in zip(self.characters, self.cpus, self.pads)]
    self.settings_mm = MenuManager(settings, pid=self.cpus[0], pad=self.pads[0])
    
    # FIXME: what do these do?
    #self._seed()
    self.reset()
    self.viewer = None

    # Just need to initialize the relevant attributes
    self._configure()
    
    self.setup()

  def write_locations(self):
    path = self.dolphin_dir + '/MemoryWatcher/'
    os.makedirs(path, exist_ok=True)
    print('Writing locations to:', path)
    with open(path + 'Locations.txt', 'w') as f:
      f.write('\n'.join(self.sm.locations()))
  
  def _close(self):
    self.dolphin_process.terminate()
    import shutil
    shutil.rmtree(self.dolphin_dir)
  
  def update_state(self):
    ctutil.copy(self.state, self.prev_state)
    messages = self.mw.get_messages()
    for message in messages:
      self.sm.handle(self.state, *message)

  def setup(self):
    self.update_state()
    
    while self.state.menu in [menu.value for menu in [Menu.Characters, Menu.Stages]]:
      self.mw.advance()
      last_frame = self.state.frame
      self.update_state()
      
      if self.state.frame > last_frame:
        # FIXME: this is very convoluted
        done = True
        for mm in self.menu_managers:
          if not mm.reached:
            done = False
          mm.move(self.state)
        
        if done:
          #print("menu_managers done")
          if self.settings_mm.reached:
            #print("playing movie")
            self.movie.play(self.pads[0])
          else:
            #print("settings_mm")
            self.settings_mm.move(self.state)
    
    print("setup finished")
    assert(self.state.menu == Menu.Game.value)

  def _seed(self, seed=None):
    from gym.utils import seeding
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def _step(self, action):
    assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
    
    self.pads[0].send_controller(self.realController(action))
    
    self.mw.advance()
    self.update_state()
    
    observation = gameConv(self.state)
    reward = computeRewards([self.prev_state, self.state])[0]

    return observation, reward, False, {}

  def _reset(self):
    return gameConv(self.state)

def simpleSmashEnv(act_every=3, **kwargs):
  env = SmashEnv(**kwargs)
  env.action_space = spaces.Discrete(len(ssbm.simpleControllerStates))
  env.realController = lambda action: ssbm.simpleControllerStates[action].realController()
  
  from gym.wrappers import SkipWrapper
  return SkipWrapper(3)(env)

