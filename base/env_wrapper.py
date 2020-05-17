from . import atari_wrappers

from collections import deque
import gym
from gym import spaces
import pybulletgym
import numpy as np


import cv2
cv2.ocl.setUseOpenCL(False)

def create_env(flags):
    if flags.ObsType == "Atari":
        return atari_wrappers.wrap_pytorch(
            atari_wrappers.wrap_deepmind(
                atari_wrappers.make_atari(flags.env),
                clip_rewards=False,
                frame_stack=flags.num_stack,
                scale=False,
                grayscale=False,
            )
        )
    elif flags.ObsType == "State":
        return FrameStack_state(gym.make(flags.env), flags.num_stack)
    elif flags.ObsType == "Image":
        
        return FrameStack_st_State2Image(gym.make(flags.env), flags.num_stack)


class FrameStack_state(gym.Wrapper):
    def __init__(self, env, k=1):
        """Stack k last frames.

        Returns lazy array, which is much more memory efficient.

        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=np.finfo(env.observation_space.dtype).min, high=np.finfo(env.observation_space.dtype).max, shape=(shp[:-1] + (shp[-1] * k,)), dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return (np.array(self.frames).reshape(-1))


class FrameStack_st_State2Image(gym.Wrapper):
    def __init__(self, env, k=1, width=84, height=84, grayscale=True):
        """Stack k last frames.

        Returns lazy array, which is much more memory efficient.

        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        #from pyvirtualdisplay import Display
        #self.display = Display(visible=0, size=(1400, 900))
        #self.display.start()

        env.render()
        gym.Wrapper.__init__(self, env)

        self._width = width
        self._height = height
        self._grayscale = grayscale
        
        if self._grayscale:
            num_colors = 1
        else:
            num_colors = 3

        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width, num_colors),
            dtype=np.uint8,
        )
        
        self.k = k
        self.frames = deque([], maxlen=k)

        shp = (self._height, self._width, num_colors)
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[:-1] + (shp[-1] * k,)), dtype=np.uint8)

    def reset(self):
        ob = self.env.reset()
        ob = self._get_rgb_array()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        ob = self._get_rgb_array()
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return np.array(self.frames).reshape(self.observation_space.shape)

    def _get_rgb_array(self):
        frame = self.env.render(mode="rgb_array")
        if self._grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self._width, self._height), interpolation=cv2.INTER_AREA
        )
        if self._grayscale:
            frame = np.expand_dims(frame, -1)
        return frame


