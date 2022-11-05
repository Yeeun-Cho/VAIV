def Body(candle):
    return abs(candle.Open - candle.Close)


def Upper_Shadow(candle):
    return candle.High - max(candle.Open, candle.Close)


def Lower_Shadow(candle):
    return min(candle.Open, candle.Close) - candle.Low


def Candle(candle):
    return candle.High - candle.Low


def Bullish_Candle(candle):
    return candle.Open < candle.Close


def Bearish_Candle(candle):
    return candle.Open > candle.Close


def Bullish_Harami(date, drange, section):
    i = drange.index(date)  # 큰 음봉의 위치
    if i < (len(drange)-1):
        c1 = section.loc[date]
        c2 = section.loc[drange[i+1]]
        condition = []
        condition.append(Bearish_Candle(c1))
        condition.append(Bullish_Candle(c2))
        condition.append(c1.Close < c2.Open)
        condition.append(c1.Open > c2.Close)
        condition.append(Body(c1) > 0.6*Candle(c1))
        return (False not in condition)
    return False


def Bullish_Engulfing(date, drange, section):
    i = drange.index(date)  # 큰 양봉의 위치
    if i < (len(drange)-1):
        c1 = section.loc[date]
        c2 = section.loc[drange[i+1]]
        condition = []
        condition.append(Bearish_Candle(c1))
        condition.append(Bullish_Candle(c2))
        condition.append(c1.Close > c2.Open)
        condition.append(c1.Open < c2.Close)
        condition.append(Body(c2) > 0.6*Candle(c2))
        return (False not in condition)
    return False


def Bullish_Doji(date, drange, section):
    i = drange.index(date)
    if i > 0:
        c1 = section.loc[drange[i-1]]
        c2 = section.loc[date]
    else:
        c1 = section.loc[date]
        c2 = section.loc[drange[i+1]]
    condition = []
    condition.append(Bearish_Candle(c1))
    condition.append(c1.Low > c2.Low)
    condition.append(Upper_Shadow(c2) > 3*Body(c2))
    condition.append(Upper_Shadow(c2) > 3*Lower_Shadow(c2))
    condition.append(Body(c1) > 0.6*Candle(c1))
    return (False not in condition)


def Hammer(date, drange, section):
    i = drange.index(date)  # 큰 음봉의 위치
    if i < (len(drange)-1):
        c1 = section.loc[date]
        c2 = section.loc[drange[i+1]]
        condition = []
        condition.append(Bearish_Candle(c1))
        condition.append(c1.Low > c2.Low)
        condition.append(Lower_Shadow(c2) > 2*Body(c2))
        condition.append(Upper_Shadow(c2) > 0.3*Body(c2))
        condition.append(Body(c1) > 0.6*Candle(c1))
        return (False not in condition)
    else:
        return False


def Morningstar(date, drange, section):
    i = drange.index(date)  # 큰 음봉의 위치
    if ((i > 0) & (i < len(drange) - 1)):
        c1 = section.loc[drange[i-1]]
        c2 = section.loc[date]
        c3 = section.loc[drange[i+1]]
        condition = []
        condition.append(Bearish_Candle(c1))
        condition.append(Bullish_Candle(c3))
        condition.append(Body(c1) > 0.6*Candle(c1))
        condition.append(c2.Open < c1.Close)
        condition.append(c3.Open > c2.Close)
        condition.append(Body(c2) < 0.3*Candle(c2))
        condition.append(Body(c2) < Body(c1))
        condition.append(Body(c2) < Body(c3))
        condition.append(c2.Low < c3.Low)
        condition.append(c2.Low < c1.Low)
        condition.append(c2.High < c1.Open)
        condition.append(c2.High < c3.Close)
        return (False not in condition)
    return False
    


def Bearish_Harami(date, drange, section):
    i = drange.index(date)  # 큰 음봉의 위치
    if i > (len(drange)-1):
        c1 = section.loc[date]
        c2 = section.loc[drange[i+1]]
        condition = []
        condition.append(Bullish_Candle(c1))
        condition.append(Bearish_Candle(c2))
        condition.append(c1.Close > c2.Open)
        condition.append(c1.Open < c2.Close)
        condition.append(Body(c1) > 0.6*Candle(c1))
        return (False not in condition)
    return False


def Bearish_Engulfing(date, drange, section):
    i = drange.index(date)  # 큰 양봉의 위치
    if i > (len(drange)-1):
        c1 = section.loc[date]
        c2 = section.loc[drange[i+1]]
        condition = []
        condition.append(Bullish_Candle(c1))
        condition.append(Bearish_Candle(c2))
        condition.append(c1.Close < c2.Open)
        condition.append(c1.Open > c2.Close)
        condition.append(Body(c1) > 0.6*Candle(c1))
        return (False not in condition)
    return False


def Gravestone_Doji(date, drange, section):
    i = drange.index(date)  # 큰 양봉의 위치
    if i > 0:
        c1 = section.loc[drange[i-1]]
        c2 = section.loc[date]
    else:
        c1 = section.loc[date]
        c2 = section.loc[drange[i+1]]

    condition = []
    condition.append(Bullish_Candle(c1))
    condition.append(c1.High < c2.High)
    condition.append(Upper_Shadow(c2) > 3*Body(c2))
    condition.append(Upper_Shadow(c2) > 3*Lower_Shadow(c2))
    condition.append(Body(c1) > 0.6*Candle(c1))
    return (False not in condition)


def HangingMan(date, drange, section):
    i = drange.index(date)  # 큰 음봉의 위치
    if i > 0:
        c1 = section.loc[drange[i-1]]
        c2 = section.loc[date]
        condition = []
        condition.append(Bullish_Candle(c1))
        condition.append(c1.High < c2.High)
        condition.append(Lower_Shadow(c2) > 2*Body(c2))
        condition.append(Upper_Shadow(c2) > 0.3*Body(c2))
        condition.append(Body(c1) > 0.6*Candle(c1))
        return (False not in condition)
    return False


def Eveningstar(date, drange, section):
    i = drange.index(date)
    if (i > 0) & (i < len(drange) - 1):
        c1 = section.loc[drange[i-1]]
        c2 = section.loc[date]
        c3 = section.loc[drange[i+1]]
        condition = []
        condition.append(Bullish_Candle(c1))
        condition.append(Bearish_Candle(c3))
        condition.append(Body(c1) > 0.6*Candle(c1))
        condition.append(c2.Open > c1.Close)
        condition.append(c3.Open < c2.Close)
        condition.append(Body(c2) < 0.3*Candle(c2))
        condition.append(Body(c2) < Body(c1))
        condition.append(Body(c2) < Body(c3))
        condition.append(c2.High > c3.High)
        condition.append(c2.High > c1.High)
        condition.append(c2.Low > c1.Open)
        condition.append(c2.Low > c3.Close)
        return (False not in condition)
    return False


def Bullish(date, drange, section):  # Buy
    pattern = ['Bullish Harami', 'Bullish Engulfing', 'Bullish_Doji', 'Hammer', 'Morningstar']
    condition = []
    condition.append(Bullish_Harami(date, drange, section))
    condition.append(Bullish_Engulfing(date, drange, section))
    condition.append(Bullish_Doji(date, drange, section))
    condition.append(Hammer(date, drange, section))
    condition.append(Morningstar(date, drange, section))
    result = {pattern[i]: condition[i] for i in range(5)}
    return result


def Bearish(date, drange, section):  # Sell
    pattern = ['Bearish Harami', 'Bearish Engulfing', 'Gravestone Doji', 'Hanging Man', 'Eveningstar']
    condition = []
    condition.append(Bearish_Harami(date, drange, section))
    condition.append(Bearish_Engulfing(date, drange, section))
    condition.append(Gravestone_Doji(date, drange, section))
    condition.append(HangingMan(date, drange, section))
    condition.append(Eveningstar(date, drange, section))
    result = {pattern[i]: condition[i] for i in range(5)}
    return result
