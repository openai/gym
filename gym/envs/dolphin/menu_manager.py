import math

from gym.envs.dolphin.pad import *

characters = dict(
  fox = (-23.5, 11.5),
  falcon = (18, 18),
  roy = (18, 5),
  marth = (11, 5),
  zelda = (11, 11),
)

settings = (0, 24)

#stages = dict()

def press(state, pad, target, cursor):
  dx = target[0] - cursor[0]
  dy = target[1] - cursor[1]
  mag = math.sqrt(dx * dx + dy * dy)
  if mag < 0.3:
      pad.press_button(Button.A)
      pad.tilt_stick(Stick.MAIN, 0.5, 0.5)
      return True
  else:
      pad.tilt_stick(Stick.MAIN, 0.5 * (dx / (mag+1)) + 0.5, 0.5 * (dy / (mag+1)) + 0.5)
      return False

class MenuManager:
    def __init__(self, target, pad=None, pid=1):
        self.target = target
        self.pad = pad
        self.pid = pid
        self.reached = False

    def move(self, state):
        if self.reached:
            # Release buttons
            self.pad.release_button(Button.A)
            #pad.tilt_stick(Stick.MAIN, 0.5, 0.5)
        else:
            player = state.players[self.pid]
            self.reached = press(state, self.pad, self.target, (player.cursor_x, player.cursor_y))

