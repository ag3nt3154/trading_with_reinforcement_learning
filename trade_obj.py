import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.misc import *

class trade:
    def __init__(self, entry_date, entry_price, quantity, fees=0, type=None):
        self.entry_date = [entry_date]
        self.entry_price = [entry_price]
        self.quantity = quantity
        self.fees = fees
        self.is_open = True
        self.type = type
        self.cost = entry_price * quantity
        self.cost_basis = self.cost / quantity
        # print(self.entry_price, self.cost_basis, self.cost)
    

    def add(self, entry_date, entry_price, quantity, fees=0):
        self.entry_date.append(entry_date)
        self.entry_price.append(entry_price)
        self.cost += entry_price * quantity
        self.quantity += quantity
        self.cost_basis = self.cost / self.quantity
        

    def close(self, exit_date, exit_price):
        self.exit_date = exit_date
        self.exit_price = exit_price
        self.is_open = False


    def calculate_stats(self, df):
        self.profit = self.quantity * (self.exit_price - self.cost_basis) - self.fees
        self.returns = self.profit / abs(self.cost)
        self.time_in_trade = self.exit_date - self.entry_date[0]
        self.capital_at_risk = abs(self.cost)
        
        mask = (df.index >= self.entry_date[0]) & (df.index <= self.exit_date)
        self.df = df.loc[mask]
        
        # max drawdown
        # potential profit
    



def generate_record(trade_list):
    df = {
        'profit': np.array([f.profit for f in trade_list]),
        'returns': np.array([f.returns for f  in trade_list]),
        'time_in_trade': np.array([f.time_in_trade.days for f in trade_list]),
        'entry_date': np.array([f.entry_date[0] for f in trade_list]),
        'exit_date': np.array([f.exit_date for f in trade_list]),
        'quantity': np.array([f.quantity for f in trade_list]),
        'capital_at_risk': np.array([f.capital_at_risk for f in trade_list])
    }
    df = pd.DataFrame.from_dict(df)

    return df





class tradeList:
    def __init__(self):
        self.list = []

    def append(self, trade):
        self.list.append(trade)

    def analyse(self, df):
        self.list = [f for f in self.list if f.is_open==False]

        for trade in self.list:
            trade.calculate_stats(df)
        
        self.win_list = [f for f in self.list if f.profit > 0]
        self.loss_list = [f for f in self.list if f.profit <= 0]

        self.records = generate_record(self.list)
        self.win_records = generate_record(self.win_list)
        self.loss_records = generate_record(self.loss_list)

        self.stats = {
            'num_wins': len(self.win_list),
            'num_loss': len(self.loss_list),
            'win_rate': len(self.win_list) / (len(self.win_list) + len(self.loss_list)),
            'profit_total': self.records['profit'].sum(),
            'profit_mean': self.records['profit'].mean(),
            'returns_mean': self.records['returns'].mean(),
            'returns_std': self.records['returns'].std(),
            'time_in_trade_mean': self.records['time_in_trade'].mean(),
            'time_in_trade_std': self.records['time_in_trade'].std(),
            'capital_at_risk_mean': self.records['capital_at_risk'].mean(),
            'win_returns_mean': self.win_records['returns'].mean(),
            'win_returns_std': self.win_records['returns'].std(),
            'win_profit_mean': self.win_records['profit'].mean(),
            'win_time_in_trade_mean': self.win_records['time_in_trade'].mean(),
            'win_time_in_trade_std': self.win_records['time_in_trade'].std(),
            'loss_returns_mean': self.loss_records['returns'].mean(),
            'loss_returns_std': self.loss_records['returns'].std(),
            'loss_profit_mean': self.loss_records['profit'].mean(),
            'loss_time_in_trade_mean': self.loss_records['time_in_trade'].mean(),
            'loss_time_in_trade_std': self.loss_records['time_in_trade'].std(),
        }

        self.records = pd.DataFrame.from_dict(self.records)
        self.win_records = pd.DataFrame.from_dict(self.win_records)
        self.loss_records = pd.DataFrame.from_dict(self.loss_records)


    def plot_graphs(self):
        fig, axs = plt.subplots(3, 2, figsize=(16, 14))
        axs[0, 0].hist(self.records['returns'], bins=20)
        axs[0, 0].set_title('Returns')
        # axs[0, 0].legend()

        axs[0, 1].hist(self.win_records['returns'], bins=20, label='win')
        axs[0, 1].hist(self.loss_records['returns'], bins=20, label='loss')
        axs[0, 1].set_title('Returns')
        axs[0, 1].legend()


        axs[1, 0].hist(self.records['time_in_trade'], bins=20)
        axs[1, 0].set_title('Time in trade')
        # axs[1, 0].legend()

        axs[1, 1].hist(self.win_records['time_in_trade'], bins=20, label='win')
        axs[1, 1].hist(self.loss_records['time_in_trade'], bins=20, label='loss')
        axs[1, 1].set_title('Time in trade')
        axs[1, 1].legend()


        axs[2, 0].hist(self.records['entry_date'], bins=20)
        axs[2, 0].set_title('Entry date')
        # axs[2, 0].legend()

        axs[2, 1].hist(self.win_records['entry_date'], bins=20, label='win')
        axs[2, 1].hist(self.loss_records['entry_date'], bins=20, label='loss')
        axs[2, 1].set_title('Entry date')
        axs[2, 1].legend()



    def plot_all_trades(self):
        for i in range(len(self.list)):
            t = self.list[i]

            print(f'trade {i}')
            print(f'entry: {t.entry_price}')
            print(f'cost_basis: {t.cost_basis}')
            print(f'exit: {t.exit_price}')
            print(f'returns: {t.returns}')
            
            plot_candle(t.df)
            
            try:
                plt.plot(t.df['stop_loss'], color='C1', label='stop loss')
            except KeyError:
                pass
            try:
                plt.scatter(t.df.index, t.df['entry_price'], marker='^', color='black', label='entry')
                plt.scatter(t.df.index, t.df['exit_price'], marker='v', color='black', label='exit')
            except KeyError:
                pass
            plt.legend()
            plt.show()
    
    
    def plot_time_period(self, df, start_datetime, end_datetime):
        mask = (df.index >= start_datetime) & (df.index <= end_datetime)
        df1 = df.loc[mask]
            
        plt.figure(figsize=(15,10)) 
        plot_candle(df1)

        try:
            plt.plot(df1['stop_loss'], color='C1', label='stop loss')
            plt.scatter(df1.index, df1['entry_price'], marker='^', color='black', label='entry')
            plt.scatter(df1.index, df1['exit_price'], marker='v', color='black', label='exit')
        except KeyError:
            pass
        plt.legend()
        plt.show()



        
    


    