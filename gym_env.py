import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from utils.misc import get_attr


class TradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df, **kwargs):
        super(TradingEnv, self).__init__()
        
        # Define intital capital
        self.initial_capital = get_attr(kwargs, 'initial_capital', 1000000)

        # define input_feature_list
        # set input_feature_list -> all the signals to be input into the model
        self.input_feature_list = get_attr(kwargs, 'input_feature_list', None)
        
        # Define state and action space
        self.state_size = get_attr(kwargs, 'state_size', 16)
        self.action_size = get_attr(kwargs, 'action_size', 9)

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

        # set dataframe
        self.df = df.copy()
        # check that volatility is present as it is required to generate the order price
        assert 'volatility' in self.input_feature_list
        # check that all input features are present in the dataframe
        for f in self.input_feature_list:
            if f not in list(self.df):
                raise Exception('Input feature ' + f + 'not found in dataframe')
        # convert to numpy array to speed up execution
        self.input_df = self.df[self.input_feature_list].to_numpy()

        # create default state of the environment
        self.clean_slate()

        # action_space = limit_order = [order_price, order_quantity]
        self.action_space = spaces.Box(
            low=np.array([0, 0]),
            high=np.array([self.action_size, self.action_size]),
            dtype=np.float32
        )
        # observation_space = [input_features, trader_state]
        self.observation_space = spaces.Box(
            low=np.array([-np.inf for f in range(self.state_size)]),
            high=np.array([np.inf for f in range(self.state_size)]),
            dtype=np.float64
        )

        # valuation price -> price used to mark the portfolio to market and calculate returns
        self.eval_price = self.df['close'].to_numpy()

        # convert OHLC prices to numpy array
        self.open_price = self.df['open'].to_numpy()
        self.close_price = self.df['close'].to_numpy()
        self.high_price = self.df['high'].to_numpy()
        self.low_price = self.df['low'].to_numpy()
        

    def clean_slate(self):
        '''
        Set initial default state of environment.
        Called by __init__ and reset
        '''
        # State variables
        self.cash = self.initial_capital
        self.position = 0
        self.position_value = 0
        self.portfolio_value = self.cash + self.position_value
        self.leverage = abs(self.position_value / self.portfolio_value)
        self.portfolio_volatility = 0
        self.portfolio_return = 1
        self.end = False
        self.current_step = 0

        # record all instantaneous values of state
        self.record = {
            'cash': [],
            'position': [],
            'position_value': [],
            'portfolio_value': [],
            'leverage': [],
            'portfolio_volatility': [],
            'portfolio_return': [],
        }

        # record all trades
        self.trade_record = {
            'order_price': [],
            'order_quantity': [],
            'execution_price': [],
            'execution_quantity': [],
            # 'cost_basis': [],
        }

        # instantaneous state of the trader
        self.trader_state = np.array([
            self.cash,
            self.position,
            self.position_value,
            self.portfolio_value,
            # self.leverage,
            # self.portfolio_volatility,
        ])


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
        obs = np.concatenate([self.input_df[self.current_step], self.trader_state])

        # update current volatility
        self.curr_volatility = self.df['volatility'].to_numpy()[self.current_step]
        # update current price
        self.curr_price = self.df['close'].to_numpy()[self.current_step]
        return obs
    

    def step(self, action):
        '''
        env takes 1 step given action input
        '''
        # define terminate condition
        if self.current_step >= len(self.input_df) - 2:
            self.end = True

        # if not terminated -> take step as described by action input
        if not self.end:
            reward = self.__take_action(action)
            obs = self.__next_observation()
            return obs, reward, self.end, {}
        
        # termination
        else:
            self.eval_render()
            obs = self.__next_observation()
            reward = 0
            return obs, reward, self.end, {}

        
    def __take_action(self, action):
        '''
        Do stuff as prescribed by action input
        '''
        # check that action input is within the action_size and is an odd number
        

        # action input value limit
        act_lim = (self.action_size - 1) // 2

        # centre action input value about 0
        action[0] -= act_lim
        action[1] -= act_lim

        assert -act_lim <= action[0] <= act_lim
        assert -act_lim <= action[1] <= act_lim

        # preset order_price, order_quantity, execution_price
        # execution_price may differ from the order_price based on the next period's market prices
        order_price = 0
        order_quantity = 0
        execution_price = 0
        execution_quantity = 0
        
        # if order_quantity is not 0 -> we are buying or selling
        if action[1] != 0:
            # action[0] defines the deviation of order price from current price in terms of std dev
            # order_price = current price + action[0] * std_dev
            order_price = self.curr_price * (1 + (action[0] / act_lim) * self.curr_volatility)


            # action[1] defines the order relative to the max long position and max short position
            # max positions can be defined by leverage later

            # max long position -> buy asset with portfolio value
            max_long_position = self.portfolio_value // self.curr_price
            # max long order -> order quantity require to achieve max long position from current position
            max_long_order_quantity = max_long_position - self.position

            # max short position -> short assets to with portfolio value as collateral
            max_short_position = -self.portfolio_value // self.curr_price
            # max short order -> order quantity required to achieve max short position from current position
            max_short_order_quantity = max_short_position - self.position
            
            # define order quantity relative to max long order and max short order
            if action[1] > 0:
                order_quantity = (action[1] / act_lim) * max_long_order_quantity
                
            else:
                order_quantity = -(action[1] / act_lim) * max_short_order_quantity
                
            # determine execution price assuming that order is a limit order placed 
            # before open for next time period, e.g. place limit order during after-market

            # if buying
            if order_quantity > 0:

                # if open < order price, we will get filled at open
                if self.open_price[self.current_step + 1] < order_price:
                    execution_price = self.open_price[self.current_step + 1]
                    execution_quantity = order_quantity

                # if low <= order price, we will get filled at order price
                elif self.low_price[self.current_step + 1] <= order_price:
                    execution_price = order_price
                    execution_quantity = order_quantity

                # if low > order price, we will not get filled
                else:
                    # execution price and quantity remains 0
                    pass
            
            # if selling
            if order_quantity < 0:
                
                # if open > order price, we will get filled at open
                if self.open_price[self.current_step + 1] > order_price:
                    execution_price = self.open_price[self.current_step + 1]
                    execution_quantity = order_quantity
                
                # if high >= order price, we will get filled at order price
                elif self.high_price[self.current_step + 1] >= order_price:
                    execution_price = order_price
                    execution_quantity = order_quantity
                
                # if high < order price, we will not get filled
                else:
                    # execution price and quantity remains 0
                    pass
            
            # # market order at close
            # execution_quantity = order_quantity
            # execution_price = self.close_price[self.current_step]

            if execution_quantity != 0:
                # change cash and position from order execution
                self.cash -= execution_price * execution_quantity
                self.position += execution_quantity
        
        # calculate new trader state
        self.position_value = self.position * self.eval_price[self.current_step]
        self.portfolio_value = self.cash + self.position_value
        self.leverage = abs(self.position_value / self.portfolio_value)
        self.portfolio_return = self.portfolio_value / self.initial_capital
        self.portfolio_volatility = np.std(
            self.record['portfolio_return'][-self.lookback_period:]
            # self.record['portfolio_return']
        )
        if self.portfolio_volatility == np.nan:
            self.portfolio_volatility = 0

        self.trader_state = np.array([
            self.cash,
            self.position,
            self.position_value,
            self.portfolio_value,
            # self.leverage,
            # self.portfolio_volatility,
        ])

        # record
        self.record['cash'].append(self.cash)
        self.record['position'].append(self.position)
        self.record['position_value'].append(self.position_value)
        self.record['portfolio_value'].append(self.portfolio_value)
        self.record['leverage'].append(self.leverage)
        self.record['portfolio_volatility'].append(self.portfolio_volatility)
        self.record['portfolio_return'].append(self.portfolio_return)
        
        self.trade_record['order_price'].append(order_price)
        self.trade_record['order_quantity'].append(order_price)
        self.trade_record['execution_price'].append(execution_price)
        self.trade_record['execution_quantity'].append(execution_quantity)
        
        reward = self.position * (
            (self.eval_price[self.current_step] - self.eval_price[self.current_step - 1]) 
            + self.hindsight_weight * (self.eval_price[self.current_step + 1] - self.eval_price[self.current_step - 1])
            )
        
        self.current_step += 1

        return reward

        

    def eval_render(self, mode='human', close=False):
        '''
        evaluate and render peformance indices
        '''
        self.total_return = self.record['portfolio_return'][-1] - 1
        self.volatility = np.std(np.array(self.record['portfolio_return']) - 1) * np.sqrt(len(self.record['portfolio_return']))
        self.sharpe = self.total_return / self.volatility
        curr_max_return = 1
        self.max_drawdown = 0
        for i in range(len(self.record['portfolio_return'])):
            if self.record['portfolio_return'][i] > curr_max_return:
                curr_max_return = self.record['portfolio_return'][i]
            else:
                drawdown = (curr_max_return - self.record['portfolio_return'][i]) / curr_max_return
                if drawdown > self.max_drawdown:
                    self.max_drawdown = drawdown
        self.calmar = self.total_return / self.max_drawdown

        if self.display:
            print(f'total return: {self.total_return}')
            print(f'volatility  : {self.volatility}')
            print(f'sharpe ratio: {self.sharpe}')
            print(f'max drawdown: {self.max_drawdown}')
            print(f'calmar ratio: {self.calmar}')
        
        
