import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from enum import Enum
import pandas as pd

class Actions(Enum):
    Sell = 0
    Buy = 1

class TradingEnv(gym.Env):
    '''
    designed for panel data 
    '''
    metadata = {'render.modes': ['human']}
    def __init__(self, df):
        assert df.ndim == 2
        self.df = df
        self.shape=(df.shape[1],)
        self.codes=df.stock_codes.uniqe()

        # spaces
        self.action_space = spaces.Discrete(len(Actions))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float32)

        self._stock=np.random.choice(self.codes)
        self._start_tick = 0
        self._ret, self._features = self._process_data(self._stock) 
        #take stock code as the only argument and return ret and features for the stock
        self._end_tick = len(self._ret) - 1
        self._done = None
        self._current_tick = None
        self._last_trade_tick = None
        self._position = None
        self._last_position = None

    def reset(self):
        #epodisode intializaton
        self._stock=np.random.choice(self.codes)
        self._start_tick = 0
        self._ret, self._features = self._process_data(self._stock)
        self._end_tick = len(self._ret) - 1
        self._done = False
        self._current_tick = self._start_tick
        self._last_trade_tick = self._current_tick - 1
        self._position = None
        self._last_position = None
        self._total_reward=0
        return self._get_observation()


    def step(self, action):
        self._done = False

        self._current_tick += 1
        if self._current_tick == self._end_tick:
            self._done = True

        step_reward = self._calculat_reward(action)
        self._total_reward += step_reward
        observation = self._get_observation()
        info = dict(
            total_reward = self._total_reward,
        )
        return observation, step_reward, self._done, info

    def _get_observation(self):
        return self._features[self._current_tick]

    def _process_data(self,stock):
        mask=self.df['stock_codes']==stock
        return self.df.loc[mask,'ret'],self.df.loc[mask, self.df.columns != 'ret']

    def _calculate_reward(self, action):
        return self._ret[self._current_tick] if action==1 else 0
