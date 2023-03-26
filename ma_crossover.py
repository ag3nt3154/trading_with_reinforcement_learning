class ma_crossover_policy:
    def __init__(self):
        pass
    def learn(self, timesteps):
        pass
    def predict(self, obs, deterministic=False, **kwargs):
        '''
        input_arr = ADJC, S, S+1
        trader_state = cash, position, position_value, portfolio_value, margin
        '''
        buy_threshold = get_attr(kwargs, 'buy_threshold', 0)
        sell_threshold = get_attr(kwargs, 'sell_threshold', 0)
        cash, position, _, _, _, _ = obs[-6:]
        adjclose, signal, signal_1 = obs[:3]

        # print(cash, position)
        

        limit_order = np.zeros(2)
        if  signal > buy_threshold and signal_1 < buy_threshold:
            limit_order[0] = adjclose
            limit_order[1] = cash // adjclose
        elif signal < sell_threshold and signal_1 > sell_threshold:
            limit_order[0] = adjclose
            limit_order[1] = -position
        # print(limit_order)
        return limit_order, None

def ma_signal(df, **kwargs):
    
    short_period = get_attr(kwargs, 'short_period', 10)
    long_period = get_attr(kwargs, 'long_period', 20)

    df['signal'] = df['adjclose'].rolling(short_period).mean() - df['adjclose'].rolling(long_period).mean()
    df['signal+1'] = df['signal'].shift(1)
    df = df.dropna()

    return df