import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from utils.misc import get_attr
from simple_backtester import backTester


class TradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    
    def __init__(self, df, **kwargs):
        super(TradingEnv, self).__init__()

        # Define intital capital
        self.initial_capital = get_attr(kwargs, 'initial_capital', 1000000)

        # define input_feature_list
        # set input_feature_list -> all the signals to be input into the model
        self.input_feature_list = get_attr(kwargs, 'input_feature_list', None)
        print(self.input_feature_list)
        # trader_state_list -> state of trader, e.g. cash, position, leverage
        self.trader_state_list = get_attr(kwargs, 'trader_state_list', None)
        
        # Define state and action space
        default_state_size = len(self.input_feature_list) + len(self.trader_state_list)
        self.state_size = get_attr(kwargs, 'state_size', default_state_size)
        self.is_discrete = get_attr(kwargs, 'is_discrete', False)
        if self.is_discrete:
            self.action_size = get_attr(kwargs, 'action_size', 9)
        else:
            self.action_size = 1

        # action input value limit
        self.act_lim = (self.action_size - 1) // 2

        # hindsight weight
        self.hindsight_weight = get_attr(kwargs, 'hindsight_weight', 0.5)

        # lookback for time-series data for input
        self.lookback_period = get_attr(kwargs, 'lookback_period', 20)
        # lookforward for hindsight bias on the reward
        self.lookforward_period = get_attr(kwargs, 'lookforward_period', 20)
        # render_window_size for visualising trades
        self.render_window_size = get_attr(kwargs, 'render_window_size', 20)
        # check render
        self.display = get_attr(kwargs, 'display', False)

        # set df
        self.df = df.copy()

        # check that all input features are present in the dataframe
        for f in self.input_feature_list:
            if f not in list(self.df):
                raise Exception('Input feature ' + f + 'not found in dataframe')

        # convert to numpy array to speed up execution
        self.input_arr = self.df[self.input_feature_list].to_numpy()

        # action_space = limit_order = [order_price, order_quantity]
        self.action_space = spaces.Box(
            low=np.array([0]),
            high=np.array([self.action_size]),
            dtype=np.float32
        )
        # observation_space = [input_features, trader_state]
        self.observation_space = spaces.Box(
            low=np.array([-np.inf for f in range(self.state_size)]),
            high=np.array([np.inf for f in range(self.state_size)]),
            dtype=np.float32
        )

        # initialize backtester and set df
        self.bt = backTester(**kwargs)
        self.bt.set_asset(self.df)
        self.trader_state = self.bt.trader_state

        # create default state of the environment
        self.clean_slate()


    def clean_slate(self):
        '''
        Set initial default state of environment.
        Called by __init__ and reset
        '''
        self.bt.clean_slate()
        self.end = False
        self.current_step = 0
        

        
    def reset(self):
        '''
        Reset env to the clean default state and return state observation
        '''
        self.clean_slate()
        return self.__next_observation()
        

    def __next_observation(self):
        '''
        return state/observation
        '''
        obs = np.concatenate([self.input_arr[self.current_step], self.trader_state])

        # # update current volatility
        # self.curr_volatility = self.df['volatility'].to_numpy()[self.current_step]
        # # update current price
        # self.curr_price = self.df['close'].to_numpy()[self.current_step]
        return obs
        

    def step(self, action):
        '''
        env takes 1 step given action input
        '''
        # define terminate condition
        if self.bt.end:
            self.end = True

        # if not terminated -> take step as described by action input
        if not self.end:
            reward = self.__take_action(action)
            obs = self.__next_observation()
            return obs, reward, self.end, {}
        
        # termination
        else:
            # self.bt.analyse()
            self.eval_render()
            obs = self.__next_observation()
            reward = 0
            return obs, reward, self.end, {}
        

    def __take_action(self, action):
        '''
        Do stuff as prescribed by action input
        '''
        
        

        # centre action input value about 0
        action -= self.act_lim
        order_quantity = 0

        if action != 0:

            # check that action input is within the action_size and is an odd number
            assert -self.act_lim <= action <= self.act_lim     

            # max long position -> buy asset with portfolio value
            max_long_position = self.bt.portfolio_value // self.bt.open[self.current_step]
            # max long order -> order quantity require to achieve max long position from current position
            max_long_order_quantity = max_long_position - self.bt.position

            # max short position -> short assets to with portfolio value as collateral
            max_short_position = -self.bt.portfolio_value // self.bt.open[self.current_step]
            # max short order -> order quantity required to achieve max short position from current position
            max_short_order_quantity = max_short_position - self.bt.position
            
            # define order quantity relative to max long order and max short order
            if action > 0:
                order_quantity = int((action / self.act_lim) * max_long_order_quantity)
            else:
                order_quantity = int(-(action / self.act_lim) * max_short_order_quantity)

        # trade in backtester with order_quantity   
        self.bt.execute_order(order_quantity=order_quantity)

        # set trader state
        self.trader_state = self.bt.trader_state
        
        # set reward function
        reward = self.bt.position * (
            (self.bt.close[self.current_step + 1] - self.bt.close[self.current_step]) 
        )
        self.current_step = self.bt.current_step

        return reward
    

    def eval_render(self, mode='human', close=False):
        '''
        evaluate and render peformance indices
        '''
        if self.display:
            print(f'annual return: {self.bt.annual_return}')
            print(f'volatility  : {self.bt.volatility}')
            print(f'sharpe ratio: {self.bt.sharpe}')




