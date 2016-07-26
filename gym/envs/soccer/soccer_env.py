import os, subprocess, time, signal
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding

try:
    import hfo_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you can install HFO dependencies with 'pip install gym[soccer].)'".format(e))

import logging
logger = logging.getLogger(__name__)

class SoccerEnv(gym.Env, utils.EzPickle):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.viewer = None
        self.server_process = None
        self.server_port = None
        self.hfo_path = hfo_py.get_hfo_path()
        self._configure_environment()
        self.env = hfo_py.HFOEnvironment()
        self.env.connectToServer(config_dir=hfo_py.get_config_path())
        self.observation_space = spaces.Box(low=-1, high=1,
                                            shape=(self.env.getStateSize()))
        # Action space omits the Tackle/Catch actions, which are useful on defense
        self.action_space = spaces.Tuple((spaces.Discrete(3),
                                          spaces.Box(low=0, high=100, shape=1),
                                          spaces.Box(low=-180, high=180, shape=1)))
        self.status = hfo_py.IN_GAME

    def __del__(self):
        self.env.act(hfo_py.QUIT)
        self.env.step()
        os.kill(self.server_process.pid, signal.SIGINT)
        if self.viewer is not None:
            os.kill(self.viewer.pid, signal.SIGKILL)

    def _configure_environment(self):
        """
        Provides a chance for subclasses to override this method and supply
        a different server configuration. By default, we initialize one
        offense agent against no defenders.
        """
        self._start_hfo_server()

    def _start_hfo_server(self, headless=True, frames_per_trial=500,
                          untouched_time=100, offense_agents=1,
                          defense_agents=0, offense_npcs=0,
                          defense_npcs=0, offense_team="base",
                          defense_team="base", no_sync=False,
                          port=6000, record=False, offense_on_ball=0,
                          fullstate=False, seed=-1, message_size=1000,
                          ball_x_min=0.0, ball_x_max=0.2,
                          verbose=False, log_dir="log"):
        """
        Starts the Half-Field-Offense server.
        headless: Run without a visual display.
        frames_per_trial: Trials end after this many steps.
        untouched_time: Trials end if the ball is untouched for this many steps.
        offense_agents: Number of user-controlled offensive players.
        defense_agents: Number of user-controlled defenders.
        offense_npcs: Number of offensive bots.
        defense_npcs: Number of defense bots.
        offense_team: Policy that offensive bots should use: either base/helios.
        defense_team: Policy that defense bots should use: either base/helios.
        no_sync: Disable sync mode, and run in real time. Slow!
        port: Port to start the server on.
        record: Enable recording of states & actions from all players.
        offense_on_ball: Player to give the ball to at beginning of episode.
        fullstate: Enable noise-free perception.
        seed: Seed the starting positions of the players and ball.
        message_size: Buffer size for spoken communication.
        ball_x_[min/max]: Initialize the ball this far downfield: [0,1]
        verbose: Don't suppress server messages.
        log_dir: Directory to place game logs (*.rcg).
        """
        self.server_port = port
        cmd = self.hfo_path + \
              " --frames-per-trial %i --untouched-time %i" \
              " --offense-agents %i --defense-agents %i --offense-npcs %i"\
              " --defense-npcs %i --offense-team %s --defense-team %s --port %i"\
              " --offense-on-ball %i --seed %i --message-size %i --ball-x-min %f"\
              " --ball-x-max %f --log-dir %s"\
              % (frames_per_trial, untouched_time, offense_agents,
                 defense_agents, offense_npcs, defense_npcs, offense_team, defense_team,
                 port, offense_on_ball, seed, message_size, ball_x_min, ball_x_max, log_dir)
        if headless: cmd += " --headless"
        if no_sync: cmd += " --no-sync"
        if record: cmd += " --record"
        if fullstate: cmd += " --fullstate"
        if verbose: cmd += " --verbose"
        print "Starting server with command: %s" % cmd
        self.server_process = subprocess.Popen(cmd.split(' '), shell=False)
        time.sleep(10) # Wait for server to startup before connecting a player

    def _start_viewer(self):
        """
        Starts the SoccerWindow visualizer. Note the viewer may also be
        used with a *.rcg logfile to replay a game. See details at
        https://github.com/LARG/HFO/blob/master/doc/manual.pdf.
        """
        cmd = hfo_py.get_viewer_path() +\
              " --connect --port %d" % (self.server_port)
        self.viewer = subprocess.Popen(cmd.split(' '), shell=False)

    def _step(self, action):
        self._take_action(action)
        self.status = self.env.step()
        reward = self._get_reward()
        ob = self.env.getState()
        episode_over = self.status != hfo_py.IN_GAME
        return ob, reward, episode_over, {}

    def _take_action(self, action):
        """ Converts the action space into an HFO action. """
        action_type = ACTION_LOOKUP[action[0]]
        power = action[1]
        direction = action[2]
        if action_type == hfo_py.DASH:
            self.env.act(action_type, power, direction)
        elif action_type == hfo_py.TURN:
            self.env.act(action_type, direction)
        elif action_type == hfo_py.KICK:
            self.env.act(action_type, power, direction)
        elif action_type == hfo_py.TACKLE:
            self.env.act(action_type, direction)
        elif action_type == hfo_py.CATCH:
            self.env.act(action_type)
        else:
            print "Unrecognized action %d" % action_type
            self.env.act(hfo_py.NOOP)

    def _get_reward(self):
        """ Reward is given for scoring a goal. """
        if self.status == hfo_py.GOAL:
            return 1
        else:
            return 0

    def _reset(self):
        """ Repeats NO-OP action until a new episode begins. """
        while self.status == hfo_py.IN_GAME:
            self.env.act(hfo_py.NOOP)
            self.status = self.env.step()
        while self.status != hfo_py.IN_GAME:
            self.env.act(hfo_py.NOOP)
            self.status = self.env.step()
        return self.env.getState()

    def _render(self, mode='human', close=False):
        """ Viewer only supports human mode currently. """
        if close:
            if self.viewer is not None:
                os.kill(self.viewer.pid, signal.SIGKILL)
        else:
            if self.viewer is None:
                self._start_viewer()

ACTION_LOOKUP = {
    0 : hfo_py.DASH,
    1 : hfo_py.TURN,
    2 : hfo_py.KICK,
    3 : hfo_py.TACKLE, # Used on defense to slide tackle the ball
    4 : hfo_py.CATCH,  # Used only by goalie to catch the ball
}
