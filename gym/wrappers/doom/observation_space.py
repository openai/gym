import gym
from gym import utils, spaces

try:
    from doom_py import ScreenResolution
except ImportError as e:
    raise gym.error.DependencyNotInstalled("{}. (HINT: you can install Doom dependencies " +
                                           "with 'pip install gym[doom].)'".format(e))

__all__ = ['Res160x120', 'Res200x125', 'Res200x150', 'Res256x144', 'Res256x160', 'Res256x192', 'Res320x180', 'Res320x200',
            'Res320x240', 'Res320x256', 'Res400x225', 'Res400x250', 'Res400x300', 'Res512x288', 'Res512x320', 'Res512x384',
            'Res640x360', 'Res640x400', 'Res640x480', 'Res800x450', 'Res800x500', 'Res800x600', 'Res1024x576', 'Res1024x640',
            'Res1024x768', 'Res1280x720', 'Res1280x800', 'Res1280x960', 'Res1280x1024', 'Res1400x787', 'Res1400x875',
            'Res1400x1050', 'Res1600x900', 'Res1600x1000', 'Res1600x1200', 'Res1920x1080']

class Res160x120(gym.Wrapper):
    """ Changes observation space (screen resolution) to 160x120 """
    def __init__(self, env):
        unwrapped = self._unwrapped
        self.screen_width, self.screen_height, unwrapped.screen_resolution = 160, 120, ScreenResolution.RES_160X120
        unwrapped.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_height, self.screen_width, 3))
        self.observation_space = unwrapped.observation_space

class Res200x125(gym.Wrapper):
    """ Changes observation space (screen resolution) to 200x125 """
    def __init__(self, env):
        unwrapped = self._unwrapped
        self.screen_width, self.screen_height, unwrapped.screen_resolution = 200, 125, ScreenResolution.RES_200X125
        unwrapped.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_height, self.screen_width, 3))
        self.observation_space = unwrapped.observation_space

class Res200x150(gym.Wrapper):
    """ Changes observation space (screen resolution) to 200x150 """
    def __init__(self, env):
        unwrapped = self._unwrapped
        self.screen_width, self.screen_height, unwrapped.screen_resolution = 200, 150, ScreenResolution.RES_200X150
        unwrapped.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_height, self.screen_width, 3))
        self.observation_space = unwrapped.observation_space

class Res256x144(gym.Wrapper):
    """ Changes observation space (screen resolution) to 256x144 """
    def __init__(self, env):
        unwrapped = self._unwrapped
        self.screen_width, self.screen_height, unwrapped.screen_resolution = 256, 144, ScreenResolution.RES_256X144
        unwrapped.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_height, self.screen_width, 3))
        self.observation_space = unwrapped.observation_space

class Res256x160(gym.Wrapper):
    """ Changes observation space (screen resolution) to 256x160 """
    def __init__(self, env):
        unwrapped = self._unwrapped
        self.screen_width, self.screen_height, unwrapped.screen_resolution = 256, 160, ScreenResolution.RES_256X160
        unwrapped.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_height, self.screen_width, 3))
        self.observation_space = unwrapped.observation_space

class Res256x192(gym.Wrapper):
    """ Changes observation space (screen resolution) to 256x192 """
    def __init__(self, env):
        unwrapped = self._unwrapped
        self.screen_width, self.screen_height, unwrapped.screen_resolution = 256, 192, ScreenResolution.RES_256X192
        unwrapped.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_height, self.screen_width, 3))
        self.observation_space = unwrapped.observation_space

class Res320x180(gym.Wrapper):
    """ Changes observation space (screen resolution) to 320x180 """
    def __init__(self, env):
        unwrapped = self._unwrapped
        self.screen_width, self.screen_height, unwrapped.screen_resolution = 320, 180, ScreenResolution.RES_320X180
        unwrapped.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_height, self.screen_width, 3))
        self.observation_space = unwrapped.observation_space

class Res320x200(gym.Wrapper):
    """ Changes observation space (screen resolution) to 320x200 """
    def __init__(self, env):
        unwrapped = self._unwrapped
        self.screen_width, self.screen_height, unwrapped.screen_resolution = 320, 200, ScreenResolution.RES_320X200
        unwrapped.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_height, self.screen_width, 3))
        self.observation_space = unwrapped.observation_space

class Res320x240(gym.Wrapper):
    """ Changes observation space (screen resolution) to 320x240 """
    def __init__(self, env):
        unwrapped = self._unwrapped
        self.screen_width, self.screen_height, unwrapped.screen_resolution = 320, 240, ScreenResolution.RES_320X240
        unwrapped.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_height, self.screen_width, 3))
        self.observation_space = unwrapped.observation_space

class Res320x256(gym.Wrapper):
    """ Changes observation space (screen resolution) to 320x256 """
    def __init__(self, env):
        unwrapped = self._unwrapped
        self.screen_width, self.screen_height, unwrapped.screen_resolution = 320, 256, ScreenResolution.RES_320X256
        unwrapped.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_height, self.screen_width, 3))
        self.observation_space = unwrapped.observation_space

class Res400x225(gym.Wrapper):
    """ Changes observation space (screen resolution) to 400x225 """
    def __init__(self, env):
        unwrapped = self._unwrapped
        self.screen_width, self.screen_height, unwrapped.screen_resolution = 400, 225, ScreenResolution.RES_400X225
        unwrapped.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_height, self.screen_width, 3))
        self.observation_space = unwrapped.observation_space

class Res400x250(gym.Wrapper):
    """ Changes observation space (screen resolution) to 400x250 """
    def __init__(self, env):
        unwrapped = self._unwrapped
        self.screen_width, self.screen_height, unwrapped.screen_resolution = 400, 250, ScreenResolution.RES_400X250
        unwrapped.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_height, self.screen_width, 3))
        self.observation_space = unwrapped.observation_space

class Res400x300(gym.Wrapper):
    """ Changes observation space (screen resolution) to 400x300 """
    def __init__(self, env):
        unwrapped = self._unwrapped
        self.screen_width, self.screen_height, unwrapped.screen_resolution = 400, 300, ScreenResolution.RES_400X300
        unwrapped.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_height, self.screen_width, 3))
        self.observation_space = unwrapped.observation_space

class Res512x288(gym.Wrapper):
    """ Changes observation space (screen resolution) to 512x288 """
    def __init__(self, env):
        unwrapped = self._unwrapped
        self.screen_width, self.screen_height, unwrapped.screen_resolution = 512, 288, ScreenResolution.RES_512X288
        unwrapped.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_height, self.screen_width, 3))
        self.observation_space = unwrapped.observation_space

class Res512x320(gym.Wrapper):
    """ Changes observation space (screen resolution) to 512x320 """
    def __init__(self, env):
        unwrapped = self._unwrapped
        self.screen_width, self.screen_height, unwrapped.screen_resolution = 512, 320, ScreenResolution.RES_512X320
        unwrapped.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_height, self.screen_width, 3))
        self.observation_space = unwrapped.observation_space

class Res512x384(gym.Wrapper):
    """ Changes observation space (screen resolution) to 512x384 """
    def __init__(self, env):
        unwrapped = self._unwrapped
        self.screen_width, self.screen_height, unwrapped.screen_resolution = 512, 384, ScreenResolution.RES_512X384
        unwrapped.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_height, self.screen_width, 3))
        self.observation_space = unwrapped.observation_space

class Res640x360(gym.Wrapper):
    """ Changes observation space (screen resolution) to 640x360 """
    def __init__(self, env):
        unwrapped = self._unwrapped
        self.screen_width, self.screen_height, unwrapped.screen_resolution = 640, 360, ScreenResolution.RES_640X360
        unwrapped.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_height, self.screen_width, 3))
        self.observation_space = unwrapped.observation_space

class Res640x400(gym.Wrapper):
    """ Changes observation space (screen resolution) to 640x400 """
    def __init__(self, env):
        unwrapped = self._unwrapped
        self.screen_width, self.screen_height, unwrapped.screen_resolution = 640, 400, ScreenResolution.RES_640X400
        unwrapped.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_height, self.screen_width, 3))
        self.observation_space = unwrapped.observation_space

class Res640x480(gym.Wrapper):
    """ Changes observation space (screen resolution) to 640x480 """
    def __init__(self, env):
        self._uploadable = True
        unwrapped = self._unwrapped
        self.screen_width, self.screen_height, unwrapped.screen_resolution = 640, 480, ScreenResolution.RES_640X480
        unwrapped.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_height, self.screen_width, 3))
        self.observation_space = unwrapped.observation_space

class Res800x450(gym.Wrapper):
    """ Changes observation space (screen resolution) to 800x450 """
    def __init__(self, env):
        unwrapped = self._unwrapped
        self.screen_width, self.screen_height, unwrapped.screen_resolution = 800, 450, ScreenResolution.RES_800X450
        unwrapped.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_height, self.screen_width, 3))
        self.observation_space = unwrapped.observation_space

class Res800x500(gym.Wrapper):
    """ Changes observation space (screen resolution) to 800x500 """
    def __init__(self, env):
        unwrapped = self._unwrapped
        self.screen_width, self.screen_height, unwrapped.screen_resolution = 800, 500, ScreenResolution.RES_800X500
        unwrapped.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_height, self.screen_width, 3))
        self.observation_space = unwrapped.observation_space

class Res800x600(gym.Wrapper):
    """ Changes observation space (screen resolution) to 800x600 """
    def __init__(self, env):
        unwrapped = self._unwrapped
        self.screen_width, self.screen_height, unwrapped.screen_resolution = 800, 600, ScreenResolution.RES_800X600
        unwrapped.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_height, self.screen_width, 3))
        self.observation_space = unwrapped.observation_space

class Res1024x576(gym.Wrapper):
    """ Changes observation space (screen resolution) to 1024x576 """
    def __init__(self, env):
        unwrapped = self._unwrapped
        self.screen_width, self.screen_height, unwrapped.screen_resolution = 1024, 576, ScreenResolution.RES_1024X576
        unwrapped.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_height, self.screen_width, 3))
        self.observation_space = unwrapped.observation_space

class Res1024x640(gym.Wrapper):
    """ Changes observation space (screen resolution) to 1024x640 """
    def __init__(self, env):
        unwrapped = self._unwrapped
        self.screen_width, self.screen_height, unwrapped.screen_resolution = 1024, 640, ScreenResolution.RES_1024X640
        unwrapped.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_height, self.screen_width, 3))
        self.observation_space = unwrapped.observation_space

class Res1024x768(gym.Wrapper):
    """ Changes observation space (screen resolution) to 1024x768 """
    def __init__(self, env):
        unwrapped = self._unwrapped
        self.screen_width, self.screen_height, unwrapped.screen_resolution = 1024, 768, ScreenResolution.RES_1024X768
        unwrapped.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_height, self.screen_width, 3))
        self.observation_space = unwrapped.observation_space

class Res1280x720(gym.Wrapper):
    """ Changes observation space (screen resolution) to 1280x720 """
    def __init__(self, env):
        unwrapped = self._unwrapped
        self.screen_width, self.screen_height, unwrapped.screen_resolution = 1280, 720, ScreenResolution.RES_1280X720
        unwrapped.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_height, self.screen_width, 3))
        self.observation_space = unwrapped.observation_space

class Res1280x800(gym.Wrapper):
    """ Changes observation space (screen resolution) to 1280x800 """
    def __init__(self, env):
        unwrapped = self._unwrapped
        self.screen_width, self.screen_height, unwrapped.screen_resolution = 1280, 800, ScreenResolution.RES_1280X800
        unwrapped.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_height, self.screen_width, 3))
        self.observation_space = unwrapped.observation_space

class Res1280x960(gym.Wrapper):
    """ Changes observation space (screen resolution) to 1280x960 """
    def __init__(self, env):
        unwrapped = self._unwrapped
        self.screen_width, self.screen_height, unwrapped.screen_resolution = 1280, 960, ScreenResolution.RES_1280X960
        unwrapped.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_height, self.screen_width, 3))
        self.observation_space = unwrapped.observation_space

class Res1280x1024(gym.Wrapper):
    """ Changes observation space (screen resolution) to 1280x1024 """
    def __init__(self, env):
        unwrapped = self._unwrapped
        self.screen_width, self.screen_height, unwrapped.screen_resolution = 1280, 1024, ScreenResolution.RES_1280X1024
        unwrapped.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_height, self.screen_width, 3))
        self.observation_space = unwrapped.observation_space

class Res1400x787(gym.Wrapper):
    """ Changes observation space (screen resolution) to 1400x747 """
    def __init__(self, env):
        unwrapped = self._unwrapped
        self.screen_width, self.screen_height, unwrapped.screen_resolution = 1400, 787, ScreenResolution.RES_1400X787
        unwrapped.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_height, self.screen_width, 3))
        self.observation_space = unwrapped.observation_space

class Res1400x875(gym.Wrapper):
    """ Changes observation space (screen resolution) to 1400x875 """
    def __init__(self, env):
        unwrapped = self._unwrapped
        self.screen_width, self.screen_height, unwrapped.screen_resolution = 1400, 875, ScreenResolution.RES_1400X875
        unwrapped.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_height, self.screen_width, 3))
        self.observation_space = unwrapped.observation_space

class Res1400x1050(gym.Wrapper):
    """ Changes observation space (screen resolution) to 1400x1050 """
    def __init__(self, env):
        unwrapped = self._unwrapped
        self.screen_width, self.screen_height, unwrapped.screen_resolution = 1400, 1050, ScreenResolution.RES_1400X1050
        unwrapped.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_height, self.screen_width, 3))
        self.observation_space = unwrapped.observation_space

class Res1600x900(gym.Wrapper):
    """ Changes observation space (screen resolution) to 1600x900 """
    def __init__(self, env):
        unwrapped = self._unwrapped
        self.screen_width, self.screen_height, unwrapped.screen_resolution = 1600, 900, ScreenResolution.RES_1600X900
        unwrapped.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_height, self.screen_width, 3))
        self.observation_space = unwrapped.observation_space

class Res1600x1000(gym.Wrapper):
    """ Changes observation space (screen resolution) to 1600x1000 """
    def __init__(self, env):
        unwrapped = self._unwrapped
        self.screen_width, self.screen_height, unwrapped.screen_resolution = 1600, 1000, ScreenResolution.RES_1600X1000
        unwrapped.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_height, self.screen_width, 3))
        self.observation_space = unwrapped.observation_space

class Res1600x1200(gym.Wrapper):
    """ Changes observation space (screen resolution) to 1600x1200 """
    def __init__(self, env):
        unwrapped = self._unwrapped
        self.screen_width, self.screen_height, unwrapped.screen_resolution = 1600, 1200, ScreenResolution.RES_1600X1200
        unwrapped.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_height, self.screen_width, 3))
        self.observation_space = unwrapped.observation_space

class Res1920x1080(gym.Wrapper):
    """ Changes observation space (screen resolution) to 1920x1080 """
    def __init__(self, env):
        unwrapped = self._unwrapped
        self.screen_width, self.screen_height, unwrapped.screen_resolution = 1920, 1080, ScreenResolution.RES_1920X1080
        unwrapped.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_height, self.screen_width, 3))
        self.observation_space = unwrapped.observation_space
