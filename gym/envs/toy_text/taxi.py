import sys
from six import StringIO
from gym import utils
from gym.envs.toy_text import discrete
import numpy as np

MAP = [
    "+---------+",
    "|R: | : :G|",
    "| : : : : |",
    "| : : : : |",
    "| | : | : |",
    "|Y| : |B: |",
    "+---------+",
]

class TaxiEnv(discrete.DiscreteEnv):
    """
    The Taxi Problem
    from "Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition"
    by Tom Dietterich, https://arxiv.org/abs/cs/9905014

    actions:
    - north
    - south
    - east
    - west
    - pickup
    - dropoff

    rendering:
    - blue: passenger
    - magenta: destination
    - yellow: empty taxi
    - green: full taxi
    - letters (R, G, Y, B): locations

    """
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self):
        self.desc = np.asarray(MAP, dtype='c')

        self.locs = locs = [(0,0), (0,4), (3,0), (4,4)]

        nS = 500
        num_rows = 5
        num_cols = 5
        max_rows = num_rows - 1
        max_cols = num_cols - 1
        isd = np.zeros(nS)
        nA = 6
        P = {s : {a : [] for a in range(nA)} for s in range(nS)}

        for row in range(num_rows):
            for col in range(num_cols):
                for passidx in range(len(locs) + 1): # +1 for being inside taxi
                    for destidx in range(len(locs)):

                        state = self.encode(row, col, passidx, destidx)
                        if passidx < 4 and passidx != destidx:
                            isd[state] += 1

                        for action in range(nA):
                            # defaults
                            newrow, newcol, newpassidx = row, col, passidx
                            reward = -1
                            done = False
                            taxiloc = (row, col)

                            if action == 0:
                                newrow = min(row + 1, max_rows)
                            elif action == 1:
                                newrow = max(row - 1, 0)
                            elif action == 2 and self.desc[1 + row, 2 * col + 2] == b":":
                                newcol = min(col + 1, max_cols)
                            elif action == 3 and self.desc[1 + row, 2 * col] == b":":
                                newcol = max(col - 1, 0)
                            elif action == 4: # pickup
                                if (passidx < 4 and taxiloc == locs[passidx]):
                                    newpassidx = 4
                                else: # passenger not at location
                                    reward = -10
                            elif action == 5: # dropoff
                                if (taxiloc == locs[destidx]) and passidx == 4:
                                    done = True
                                    reward = 20
                                elif (taxiloc in locs) and passidx == 4:
                                    newpassidx = locs.index(taxiloc)
                                else: # tried to dropoff in wrong location
                                    reward = -10
                            # else:
                                # we either hit a wall or didn't pickup/dropoff
                                # use default reward (-1)

                            newstate = self.encode(newrow, newcol, newpassidx, destidx)
                            P[state][action].append((1.0, newstate, reward, done))

        isd /= isd.sum()
        discrete.DiscreteEnv.__init__(self, nS, nA, P, isd)

    def encode(self, taxirow, taxicol, passloc, destidx):
        # (5) 5, 5, 4
        i = taxirow
        i *= 5
        i += taxicol
        i *= 5
        i += passloc
        i *= 4
        i += destidx
        return i

    def decode(self, i):
        out = []
        out.append(i % 4)
        i = i // 4
        out.append(i % 5)
        i = i // 5
        out.append(i % 5)
        i = i // 5
        out.append(i)
        assert 0 <= i < 5
        return reversed(out)

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        out = self.desc.copy().tolist()
        out = [[c.decode('utf-8') for c in line] for line in out]

        taxirow, taxicol, passidx, destidx = self.decode(self.s)

        def ul(x): return "_" if x == " " else x

        if passidx < 4:
            out[1 + taxirow][2 * taxicol + 1] = utils.colorize(out[1 + taxirow][2 * taxicol + 1], 'yellow', highlight=True)
            pi, pj = self.locs[passidx]
            out[1 + pi][2 * pj + 1] = utils.colorize(out[1 + pi][2 * pj + 1], 'blue', bold=True)
        else: # passenger in taxi
            out[1 + taxirow][2 * taxicol + 1] = utils.colorize(ul(out[1 + taxirow][2 * taxicol + 1]), 'green', highlight=True)

        di, dj = self.locs[destidx]
        out[1 + di][2 * dj + 1] = utils.colorize(out[1 + di][2 * dj + 1], 'magenta')

        outfile.write("\n".join(["".join(row) for row in out]) + "\n")

        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["North", "South", "East", "West", "Pickup", "Dropoff"][self.lastaction]))
        else: outfile.write("\n")

        # No need to return anything for human
        if mode != 'human':
            return outfile
