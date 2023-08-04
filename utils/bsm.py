import numpy as np
from scipy.stats import norm

def bsm(stock_price, strike_price, dte, volatility, rf_rate=0.03):
    '''
    Calculate the delta and prices of call/put option according to BSM formula.
    
    Input:
    - current stock price of underlying
    - strike price of contract
    - days to expiry (dte)
    - volatility (historically 17.73% for SPY)
    - risk-free interest rate (3.26% for 1 Year US Treasury Bill)
    
    Output:
    - delta of call option
    - price of call option
    - delta of put option
    - price of put option
    '''
    
    # change units for dte to years
    t = dte / 252
    
    # present value of strike price
    pv_strike = strike_price * np.exp(-rf_rate * t)
    
    # d1 and d2 variables of bsm
    d1 = (np.log(stock_price / strike_price) \
          + (rf_rate + (volatility ** 2) / 2) * t) / (volatility * np.sqrt(t))
    d2 = d1 - volatility * np.sqrt(t)
    
    # delta of call option
    call_delta = norm.cdf(d1)
    
    # price of call option
    call_price = call_delta * stock_price - norm.cdf(d2) * pv_strike
    
    # delta of put option
    put_delta = -norm.cdf(-d1)
    
    # price of put option
    put_price = put_delta * stock_price + norm.cdf(-d2) * pv_strike
    
    return call_delta, call_price, put_delta, put_price


def get_d1(strike, dte, curr_price, vol, div_yield=0, rf_rate=0):
    t = dte / 252
    d1 = (np.log(curr_price / strike) \
          + (rf_rate - div_yield + (vol ** 2) / 2) * t) / (vol * np.sqrt(t))
    return d1


def get_d2(strike, dte, curr_price, vol, div_yield=0, rf_rate=0):
    t = dte / 252
    d1 = get_d1(strike, dte, curr_price, vol, div_yield, rf_rate)
    d2 = d1 - vol * np.sqrt(t)
    return d2


def get_call_delta(strike, dte, curr_price, vol, div_yield=0, rf_rate=0):
    t = dte / 252
    d1 = get_d1(strike, dte, curr_price, vol, div_yield, rf_rate)
    return np.exp(-div_yield * t) * norm.cdf(d1)


def get_put_delta(strike, dte, curr_price, vol, div_yield=0, rf_rate=0):
    t = dte / 252
    d1 = get_d1(strike, dte, curr_price, vol, div_yield, rf_rate)
    return np.exp(-div_yield * t) * (norm.cdf(d1) - 1)


def get_call_price(strike, dte, curr_price, vol, div_yield=0, rf_rate=0):
    t = dte / 252
    d1 = get_d1(strike, dte, curr_price, vol, div_yield, rf_rate)
    d2 = get_d2(strike, dte, curr_price, vol, div_yield, rf_rate)
    call_delta = get_call_delta(strike, dte, curr_price, vol, div_yield, rf_rate)
    return curr_price * call_delta - strike * np.exp(-rf_rate * t) * norm.cdf(d2)

