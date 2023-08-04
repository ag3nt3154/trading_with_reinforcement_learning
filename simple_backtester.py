import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from trade_obj import trade, tradeList
from utils import misc

class backTester:
    def __init__(self, **kwargs) -> None:
        self.df = None
        self.initial_capital = misc.get_attr(kwargs, 'initial_capital', 1E6)
        self.per_order_fees = misc.get_attr(kwargs, 'per_order_fees', 0)
        self.per_volume_fees = misc.get_attr(kwargs, 'per_volume_fees', 0)
        
        self.clean_slate()


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
        }

        # instantaneous state of the trader
        self.trader_state = np.array([
            self.cash,
            self.position,
            self.position_value,
            self.portfolio_value,
            # self.leverage,
        ])



    def set_asset(self, df):
        '''
        Set the asset df to the backtester
        Change df columns to numpy arrays for faster computation
        '''
        self.df = df.copy()['open', 'high', 'low', 'close', 'adjclose']

        self.open = self.df['open'].to_numpy()
        # self.high = self.df['high'].to_numpy()
        # self.low = self.df['low'].to_numpy()
        self.close = self.df['close'].to_numpy()
        self.adjclose = self.df['adjclose'].to_numpy()
        self.date = self.df.index.to_list()



    def execute_order(self, order_quantity=0):
        '''
        Execute order as market on open.
        Only takes order quantity as input.
        '''
        execution_price = self.open[self.current_step]
        execution_quantity = 0
        
        if order_quantity != 0:
            # execute based on order
            execution_quantity = order_quantity
            fees = self.per_order_fees + self.per_volume_fees * abs(execution_quantity)

            self.position += execution_quantity
            self.cash -= (execution_price * execution_quantity + fees)
            
                
        # calculate new trader state
        self.position_value = self.position * self.close[self.current_step]
        self.portfolio_value = self.cash + self.position_value
        self.leverage = abs(self.position_value / self.portfolio_value)

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

        self.current_step += 1

        if self.current_step == len(self.df):
            self.end = True



    def analyse(self):
        self.records = pd.DataFrame.from_dict(self.record)
        self.records['date'] = self.date
        self.records = self.records.set_index('date')
        self.records['returns'] = self.records['portfolio_value'].pct_change()
        self.records['cum_returns'] = (self.records['portfolio_value'] / self.records['portfolio_value'].to_numpy()[0])
        self.records['drawdown'] = (self.records['cum_returns'] - self.records['cum_returns'].cummax()) / self.records['cum_returns'].cummax()
        self.records['buy_hold_returns'] = self.df['adjclose'].pct_change()
        self.records['buy_hold_cum_returns'] = self.adjclose / self.adjclose[0]
        self.records['buy_hold_drawdown'] = (self.records['buy_hold_cum_returns'] - self.records['buy_hold_cum_returns'].cummax()) / self.records['buy_hold_cum_returns'].cummax()
        
        self.time_period = len(self.df)
        self.annual_return = misc.get_annualised_returns(self.records['cum_returns'].to_numpy()[-1], self.time_period) - 1
        self.annual_vol = misc.get_annualised_vol(self.records['returns'])
        
        self.buy_hold_annual_return = misc.get_annualised_returns(self.records['buy_hold_cum_returns'].to_numpy()[-1], self.time_period) - 1
        self.buy_hold_annual_vol = misc.get_annualised_vol(self.records['buy_hold_returns'])

        self.sharpe = self.annual_return / self.annual_vol
        self.buy_hold_sharpe = self.buy_hold_annual_return / self.buy_hold_annual_vol
        
         

    def plot_graphs(self):
        fig, axs = plt.subplots(4, 2, figsize=(16, 14))
        axs[0, 0].plot(self.records['cum_returns'], label='strategy')
        axs[0, 0].set_title('Cumulative returns')
        axs[0, 0].set_yscale('log')
        axs[0, 0].legend()

        axs[0, 1].plot(self.records['buy_hold_cum_returns'], label='buy_hold', color='C1')
        axs[0, 1].plot(self.records['cum_returns'], label='strategy')
        axs[0, 1].set_title('Cumulative returns')
        axs[0, 1].set_yscale('log')
        axs[0, 1].legend()

        axs[1, 0].hist(self.records['returns'], bins=50, label='strategy')
        axs[1, 0].set_title('Daily returns')
        axs[1, 0].legend()

        axs[1, 1].hist(self.records['buy_hold_returns'], bins=50, label='buy_hold', color='C1')
        axs[1, 1].hist(self.records['returns'], bins=50, label='strategy')
        axs[1, 1].set_title('Daily returns')
        axs[1, 1].legend()
        

        axs[2, 0].plot(self.records['drawdown'], label='strategy')
        axs[2, 0].set_title('Drawdown')
        axs[2, 0].legend()

        axs[2, 1].plot(self.records['buy_hold_drawdown'], label='buy_hold', color='C1')
        axs[2, 1].plot(self.records['drawdown'], label='strategy')
        axs[2, 1].set_title('Drawdown')
        axs[2, 1].legend()

        axs[3, 0].hist(self.records['drawdown'], label='strategy', bins=50)
        axs[3, 0].set_title('Drawdown')
        axs[3, 0].legend()

        axs[3, 1].hist(self.records['buy_hold_drawdown'], label='buy_hold', color='C1', bins=50)
        axs[3, 1].hist(self.records['drawdown'], label='strategy', bins=50)
        axs[3, 1].set_title('Drawdown')
        axs[3, 1].legend()





