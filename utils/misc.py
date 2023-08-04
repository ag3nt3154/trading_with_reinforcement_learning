import pandas as pd
from datetime import datetime
import yahoo_fin.stock_info as si
import numpy as np
import matplotlib.pyplot as plt


def get_price_data(ticker):
    # Get historical price data from yahoo_fin
    df = si.get_data(ticker)
    # Change date from index to column
    df.reset_index(inplace=True)
    # Change column name to 'date'
    df = df.rename(columns={'index': 'date'})
    return(df)


def del_unnamed_col(dataframe):
    '''
    Delete any column with header 'Unnamed'
    '''
    dataframe.drop(dataframe.filter(regex='Unnamed').columns, axis=1, inplace=True)
    return dataframe

def str2date(date_string):
    '''
    Change date in string format to datetime format
    '''
    if '-' in date_string:
        datetime_object = datetime.strptime(date_string, '%Y-%m-%d')
    elif '/' in date_string:
        datetime_object = datetime.strptime(date_string, '%d/%m/%Y')
    
    return pd.to_datetime(datetime_object)

def clean_df(dataframe):
    '''
    Clean up dataframe from csv. Delete 'Unnamed' columns and convert dates from string format to datetime format.
    '''
    dataframe = del_unnamed_col(dataframe)
    dataframe.reset_index(drop=True)
    try:
        dataframe['date'] = dataframe.apply(lambda x: str2date(x['date']), axis=1)
    except KeyError:
        dataframe['DATE'] = dataframe.apply(lambda x: str2date(x['DATE']), axis=1)
    return dataframe


def date2str(datetime_obj):
    day = datetime_obj.day
    month = datetime_obj.month
    year = datetime_obj.year
    
    return str(day) + '/' + str(month) + '/' + str(year)


def get_attr(args, key=None, default_value=None):
    '''
    If args is a dict: return args[key]
    If args is an object: return args.key

    If args[key] or args.key is not found, return default value
    '''
    if isinstance(args, dict):
        return args[key] if key in args else default_value
    elif isinstance(args, object):
        return getattr(args, key, default_value) if key is not None else default_value
    

def get_annualised_returns(total_return, num_days=None, num_trading_days=None):
    '''
    Calculate annualised returns from total_return
    '''
    assert num_days != None or num_trading_days != None, 'Time period required'
    if num_days != None:
        return total_return ** (365.25 / num_days)
    else:
        return total_return ** (252 / num_trading_days)
    

def get_annualised_vol(returns_arr):
    '''
    Calculate annualised vol from returns
    '''
    return np.std(returns_arr) * np.sqrt(252)


def plot_candle(df, show=False):
    '''
    Plot candle-stick chart from dataframe.
    df must contain OHLC time series as ['open', 'high', 'low', 'close']
    show -> immediately show chart -> set as false if we are plotting something else
    '''
    #define width of candlestick elements
    width = .4
    width2 = .05

    #define up and down t.df
    up = df[df.close>=df.open].copy()
    down = df[df.close<df.open].copy()

    #define colors to use
    col1 = 'green'
    col2 = 'red'

    #plot up t.df
    plt.bar(up.index,up.close-up.open,width,bottom=up.open,color=col1)
    plt.bar(up.index,up.high-up.close,width2,bottom=up.close,color=col1)
    plt.bar(up.index,up.low-up.open,width2,bottom=up.open,color=col1)

    #plot down t.df
    plt.bar(down.index,down.close-down.open,width,bottom=down.open,color=col2)
    plt.bar(down.index,down.high-down.open,width2,bottom=down.open,color=col2)
    plt.bar(down.index,down.low-down.close,width2,bottom=down.close,color=col2)

    #rotate x-axis tick labels
    plt.xticks(rotation=45, ha='right')

    if show:
        #display candlestick chart
        plt.show()


def get_total_return(arr):
    '''
    Given price series, get total return
    '''
    return arr[-1]/arr[0] - 1


def generate_price_series_norm(mean, volatility, initial_price, num_days, leverage=None):
    returns = np.random.normal(mean, volatility, num_days)
    prices = [initial_price]
    price_series = None
    lev_price_series = None
    
    for i in range(1, num_days):
        price = prices[i-1] * (1 + returns[i])
        prices.append(price)
    price_series = np.array(prices)

    if leverage is not None:
        prices = [initial_price]
        for i in range(1, num_days):
            price = prices[i-1] * (1 + leverage * returns[i])
            prices.append(price)
        lev_price_series = prices


    return price_series, lev_price_series
    