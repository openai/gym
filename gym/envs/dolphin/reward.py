import numpy as np
from gym.envs.dolphin import util

# see https://docs.google.com/spreadsheets/d/1JX2w-r2fuvWuNgGb6D3Cs4wHQKLFegZe2jhbBuIhCG8/edit#gid=13
dyingActions = set(range(0xA))

def isDyingAction(action_state):
  return action_state in dyingActions

def isDying(player):
  return isDyingAction(player.action_state)

# players tend to be dead for many frames in a row
# here we prune all but the first frame of the death
def processDeaths(deaths):
  return np.array(util.zipWith(lambda prev, next: float((not prev) and next), deaths, deaths[1:]))

def processDamages(percents):
  return np.array(util.zipWith(lambda prev, next: max(next-prev, 0), percents, percents[1:]))

# from player 1's perspective
def computeRewards(states, enemies=[1], allies=[0], damage_ratio=0.01):
  players = enemies + allies

  deaths = {p : processDeaths([isDying(state.players[p]) for state in states]) for p in players}
  damages = {p : processDamages([state.players[p].percent for state in states]) for p in players}

  losses = {p : deaths[p] + damage_ratio * damages[p] for p in players}

  return sum(losses[p] for p in enemies) - sum(losses[p] for p in allies)

