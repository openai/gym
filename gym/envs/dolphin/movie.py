from gym.envs.dolphin.pad import *

def pushButton(button):
  return lambda pad: pad.press_button(button)

def releaseButton(button):
  return lambda pad: pad.release_button(button)

def tiltStick(stick, x, y):
  return lambda pad: pad.tilt_stick(stick, x, y)

neutral = tiltStick(Stick.MAIN, 0.5, 0.5)
left = tiltStick(Stick.MAIN, 0, 0.5)
down = tiltStick(Stick.MAIN, 0.5, 0)
up = tiltStick(Stick.MAIN, 0.5, 1)

endless_netplay_battlefield = [
  # time
  (0, left),
  
  # infinite time
  (26, down),
  (45, left),
  (70, neutral),
  
  # exit settings
  (71, pushButton(Button.START)),
  (72, releaseButton(Button.START)),
  
  # enter stage select
  (100, pushButton(Button.START)),
  (101, releaseButton(Button.START)),
  
  # pick battlefield
  (110, up),
  (112, neutral),
  
  #(60 * 60, neutral),
  
  # start game
  (130, pushButton(Button.START)),
  (131, releaseButton(Button.START)),
]

class Movie:
  def __init__(self, actions):
    self.actions = actions
    self.frame = 0
    self.index = 0
  
  def play(self, pad):
    if not self.over():
      frame, action = self.actions[self.index]
      if self.frame == frame:
        action(pad)
        self.index += 1
    self.frame += 1
  
  def over(self):
    return self.index == len(self.actions)
