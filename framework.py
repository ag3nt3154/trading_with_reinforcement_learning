import numpy as np
import pandas as pd
import tqdm
from scipy.stats import laplace, norm, gmean
import matplotlib.pyplot as plt
from misc import *

def fit_laplace(price_series):
    '''
    Find returns series from price series and
    fit returns into a Laplace distribution.

    Input: numpy array
    Output: laplace params
    '''
    print(price_series)
    returns_series = (price_series[1:] - price_series[:-1]) / price_series[:-1]
    return laplace.fit(returns_series)



def generate_stock(laplace_params, num_days=252, initial_price=400):
    '''
    Generate price series based on laplace distribution 
    '''
    loc, scale = laplace_params
    
    # generate returns from laplace distribution
    s = np.random.laplace(loc, scale, num_days)

    price_series = np.insert(np.cumprod(s + 1), 0, 1) * initial_price
    return price_series


class tester:
    def __init__(self, strategy, **kwargs):
        '''
        Takes strategy object and backtests on a defined OHLCV dataframe.
        '''
        self.strategy = strategy


    def generate_signal(self, data_df, **kwargs):
        '''
        Generate signal from data_df and
        appends the signals as extra columns in df.
        '''
        self.df = self.strategy.signal_func(data_df, **kwargs)

    

    def run_strategy(self, initial_capital, **kwargs):
        '''
        Runs strategy based on signals in self.df
        '''

        trader_state = np.array([initial_capital, 0, 0, initial_capital, 0])
        trader_state_arr = []
        order_arr = []

        # Specify what features/columns are used in the strategy
        strategy_input_features = get_attr(kwargs, 'strategy_input_features', None)
        if strategy_input_features == None:
            raise Exception('Must have strategy input features')
        input_df = self.df[strategy_input_features]

        # Specify initial trader_state, e.g. initial capital
        cash, position, position_value, portfolio_value, leverage = trader_state

        for i in range(len(self.df)):
            # Specify input arr to strategy
            input_arr = np.array(input_df.loc[input_df.index[i]])

            # Run strategy to generate limit_order
            order_price, order_quantity = self.strategy.order_func(input_arr, trader_state, **kwargs)
            order_arr.append((order_price, order_quantity))

            # Update trader_state
            cash -= order_price * order_quantity
            position += order_quantity
            position_value = position * self.df.loc[self.df.index[i], 'adjclose']
            portfolio_value = cash + position_value
            leverage = position_value / portfolio_value
            trader_state = (cash, position, position_value, portfolio_value, leverage)
            trader_state_arr.append(trader_state)

        # Record values in df
        self.df['order_price'] = [f[0] for f in order_arr]
        self.df['order_quantity'] = [f[1] for f in order_arr]
        self.df['cash'] = [f[0] for f in trader_state_arr]
        self.df['position'] = [f[1] for f in trader_state_arr]
        self.df['position_value'] = [f[2] for f in trader_state_arr]
        self.df['portfolio_value'] = [f[3] for f in trader_state_arr]
        self.df['leverage'] = [f[4] for f in trader_state_arr]

        
    def evaluate_strategy(self, **kwargs):
        '''
        Generate key performance indicators
        1. Total Returns
        2. CAGR
        3. Volatility
        4. Sharpe Ratio
        5. Max Drawdown
        6. Calmar Ratio
        '''
        # returns
        self.df['returns'] = np.insert(self.df['portfolio_value'].to_numpy()[1:] / self.df['portfolio_value'][:-1].to_numpy(), 0, 1)
        # cumulative returns
        self.df['cumulative_returns'] = np.cumprod(self.df['returns'])

        self.results = {
            'Total Return': self.df['cumulative_returns'][len(self.df) - 1],
        }

        print(self.results)


    def plot_results(self, **kwargs):
        self.df['buy_order'] = self.df['order_price']
        self.df['sell_order'] = self.df['order_price']
        self.df.loc[self.df['order_quantity'] <= 0, 'buy_order'] = None
        self.df.loc[self.df['order_quantity'] >= 0, 'sell_order'] = None

        plt.figure(figsize=(12,6))
        plt.plot(self.df['date'], self.df['buy_order'], marker='^', color='green', linestyle='None')
        plt.plot(self.df['date'], self.df['sell_order'], marker='^', color='red', linestyle='None')
        plt.plot(self.df['date'], self.df['adjclose'], color='silver')

        plt.show()

        plt.figure(figsize=(12,6))
        plt.plot(self.df['date'], self.df['portfolio_value'])
        plt.show()
        

class strategy:
    def __init__(self, signal_func, order_func):
        '''
        Strategy object should have
        1. signal_func to create trading signals from data_df
        2. order_func to generate orders from trading signals
        '''
        self.signal_func = signal_func
        self.order_func = order_func