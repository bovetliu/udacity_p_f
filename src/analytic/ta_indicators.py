from typing import Tuple
import pandas as pd


def get_normalized(in_ser: pd.Series, window_size: int, std_multiplier: int=2, name: str=None) -> pd.Series:
    """

    :param in_ser: incoming series
    :param window_size: window size
    :param std_multiplier: appeared in denominator
    :param name:

    :return:
    """
    if not isinstance(in_ser, pd.Series):
        raise TypeError("in_ser should be pandas Series")
    if window_size <= 0:
        raise ValueError("window size should be larger than 0")
    rolling_mean = get_rolling_mean(in_ser, window_size)
    rolling_std = get_rolling_std(in_ser, window_size)
    tbr = (in_ser - rolling_mean) / (rolling_std.mul(std_multiplier))
    tbr.name = '{}_NORM_{}'.format(in_ser.name, window_size) if not name else name
    return tbr


def get_rolling_std(in_ser: pd.Series, window_size: int, name: str=None) -> pd.Series:
    """Return rolling standard deviation of given values, using specified window size."""
    tbr = in_ser.rolling(window=window_size).std()
    tbr.name = '{}_STD_{}'.format(in_ser.name, window_size) if not name else name
    return tbr


def get_rolling_mean(in_ser: pd.Series, window_size: int, name: str=None) -> pd.Series:
    """Return rolling mean of given values, using specified window size."""
    if not isinstance(in_ser, pd.Series):
        raise TypeError("in_ser should be pandas Series")
    if window_size <= 0:
        raise ValueError("window size should be larger than 0")

    tbr = in_ser.rolling(window=window_size, min_periods=1).mean()
    tbr.name = '{}_MEAN_{}'.format(in_ser.name, window_size) if not name else name
    return tbr


def get_ema(in_ser: pd.Series, window_size: int, name: str=None) -> pd.Series:
    """
    get ema of a series, if name is not supplied, the to-be-returned series will have name of
    '{}_EMA_{}'.format(in_ser.name, window_size)
    :param in_ser: incoming pandas Series
    :param window_size: window size
    :param name: name of returned pandas series

    :return: a pandas Series representing one EMA
    """
    if not isinstance(in_ser, pd.Series):
        raise TypeError("in_ser should be pandas Series")
    if window_size <= 0:
        raise ValueError("window size should be larger than 0")
    tbr = in_ser.ewm(span=window_size, min_periods=0, adjust=False).mean()
    tbr.name = '{}_EMA_{}'.format(in_ser.name, window_size) if not name else name
    return tbr


# Momentum
def get_momentum(in_ser: pd.Series, window_size: int, name: str=None) -> pd.Series:
    if not isinstance(in_ser, pd.Series):
        raise TypeError("in_ser should be pandas Series")
    # if window_size <= 0:
    #     raise ValueError("window size should be larger than 0")
    moment_series = in_ser.diff(window_size)
    moment_series.name = '{}_MOMEN_{}'.format(in_ser.name, window_size) if not name else name
    return moment_series


def get_frws(in_ser: pd.Series, window_size: int, name: str=None) -> pd.Series:
    """
    calculate future return sum
    :param in_ser: incoming series
    :param window_size: window size
    :param name: explicit series name

    :return: a pandas Series showing return sum of future window
    """
    daily_return = (in_ser - in_ser.shift(1)) / in_ser.shift(1)
    daily_return_reversed = pd.Series(daily_return).reindex(daily_return.index[::-1])
    daily_return_reversed_sum = daily_return_reversed.rolling(window=window_size).sum()
    daily_return_reversed_mean = daily_return_reversed.rolling(window=window_size).mean()
    daily_return_reversed_std = daily_return_reversed.rolling(window=window_size).std()
    daily_return_reversed_sharp = daily_return_reversed_mean / daily_return_reversed_std
    tbr = daily_return_reversed_sharp.reindex(daily_return_reversed_sum.index[::-1])
    tbr.name = '{}_FRS_{}'.format(in_ser.name, window_size) if not name else name
    return tbr.shift(-1)


# MACD, MACD Signal and MACD difference
def get_macd(
        in_ser: pd.Series, n_fast: int, n_slow: int, n_signal: int=9, macd_diff_name: str=None) \
        -> Tuple[pd.Series, pd.Series, pd.Series]:
    """

    :param in_ser: incoming pandas Series
    :param n_fast: span of fast ema
    :param n_slow: span of slow ema
    :param n_signal: span of ema(fast - slow)
    :param macd_diff_name: macd_diff series name

    :return:a tuple of macd, macd_signal, macd_diff
    """
    ema_fast = get_ema(in_ser, n_fast)
    ema_slow = get_ema(in_ser, n_slow)
    macd = ema_fast - ema_slow
    macd.name = '{}_MACD_{}_{}'.format(in_ser.name, n_fast, n_slow)
    macd_signal = get_ema(macd, n_signal)
    macd_signal.name = '{}_MACD_SIGN_{}'.format(macd.name, n_signal)
    macd_diff = macd - macd_signal
    macd_diff.name = macd_diff_name if macd_diff_name else '{}_DIFF_{}'.format(macd.name, macd_signal.name)
    return macd, macd_signal, macd_diff


# Rate of Change
def get_roc(df, n):
    m_series = df['Close'].diff(n - 1)
    n_series = df['Close'].shift(n - 1)
    roc_series = pd.Series(m_series / n_series, name='ROC_' + str(n))
    df = df.join(roc_series)
    return df


# Average True Range, describing volatility
def get_atr(df, n):
    i = 0
    tr_l = [0]
    while i < df.index[-1]:
        tr = max(df.get_value(i + 1, 'High'), df.get_value(i, 'Close')) - min(df.get_value(i + 1, 'Low'),
                                                                              df.get_value(i, 'Close'))
        tr_l.append(tr)
        i = i + 1
    tr_s = pd.Series(tr_l)
    atr = pd.Series(pd.ewma(tr_s, span=n, min_periods=n), name='ATR_' + str(n))
    df = df.join(atr)
    return df


def get_relative_bbands(
        in_ser: pd.Series, window_size: int, upper_band_name: str=None, lower_band_name: str=None)\
        -> Tuple[pd.Series, pd.Series]:
    """Return relative positions in band."""
    rolling_mean = get_rolling_mean(in_ser, window_size)
    rolling_std = get_rolling_std(in_ser, window_size)

    upper_band = rolling_mean + rolling_std.mul(2)
    upper_band.name = upper_band_name if upper_band_name else '{}_UPPERBB_{}'.format(in_ser.name, window_size)

    lower_band = rolling_mean - rolling_std.mul(2)
    lower_band.name = lower_band_name if lower_band_name else '{}_LOWERBB_{}'.format(in_ser.name, window_size)
    return upper_band, lower_band


# Pivot Points, Supports and Resistances
def get_ppsr(df):
    PP = pd.Series((df['High'] + df['Low'] + df['Close']) / 3)
    R1 = pd.Series(2 * PP - df['Low'])
    S1 = pd.Series(2 * PP - df['High'])
    R2 = pd.Series(PP + df['High'] - df['Low'])
    S2 = pd.Series(PP - df['High'] + df['Low'])
    R3 = pd.Series(df['High'] + 2 * (PP - df['Low']))
    S3 = pd.Series(df['Low'] - 2 * (df['High'] - PP))
    psr = {'PP': PP, 'R1': R1, 'S1': S1, 'R2': R2, 'S2': S2, 'R3': R3, 'S3': S3}
    PSR = pd.DataFrame(psr)
    df = df.join(PSR)
    return df


# Stochastic oscillator %K
def get_stok_osci_k(df):
    SOk = pd.Series((df['Close'] - df['Low']) / (df['High'] - df['Low']), name='SO%k')
    df = df.join(SOk)
    return df


# Stochastic oscillator %D
def get_stok_osci(df, n):
    SOk = pd.Series((df['Close'] - df['Low']) / (df['High'] - df['Low']), name='SO%k')
    SOd = pd.Series(pd.ewma(SOk, span=n, min_periods=n - 1), name='SO%d_' + str(n))
    df = df.join(SOd)
    return df


# Trix
def get_trik(df, n):
    EX1 = pd.ewma(df['Close'], span=n, min_periods=n - 1)
    EX2 = pd.ewma(EX1, span=n, min_periods=n - 1)
    EX3 = pd.ewma(EX2, span=n, min_periods=n - 1)
    i = 0
    ROC_l = [0]
    while i + 1 <= df.index[-1]:
        ROC = (EX3[i + 1] - EX3[i]) / EX3[i]
        ROC_l.append(ROC)
        i = i + 1
    Trix = pd.Series(ROC_l, name='Trix_' + str(n))
    df = df.join(Trix)
    return df


# Average Directional Movement Index
def get_adx(df, n, n_ADX):
    i = 0
    UpI = []
    DoI = []
    while i + 1 <= df.index[-1]:
        UpMove = df.get_value(i + 1, 'High') - df.get_value(i, 'High')
        DoMove = df.get_value(i, 'Low') - df.get_value(i + 1, 'Low')
        if UpMove > DoMove and UpMove > 0:
            UpD = UpMove
        else:
            UpD = 0
        UpI.append(UpD)
        if DoMove > UpMove and DoMove > 0:
            DoD = DoMove
        else:
            DoD = 0
        DoI.append(DoD)
        i = i + 1
    i = 0
    TR_l = [0]
    while i < df.index[-1]:
        TR = max(df.get_value(i + 1, 'High'), df.get_value(i, 'Close')) - min(df.get_value(i + 1, 'Low'),
                                                                              df.get_value(i, 'Close'))
        TR_l.append(TR)
        i = i + 1
    TR_s = pd.Series(TR_l)
    ATR = pd.Series(pd.ewma(TR_s, span=n, min_periods=n))
    UpI = pd.Series(UpI)
    DoI = pd.Series(DoI)
    PosDI = pd.Series(pd.ewma(UpI, span=n, min_periods=n - 1) / ATR)
    NegDI = pd.Series(pd.ewma(DoI, span=n, min_periods=n - 1) / ATR)
    ADX = pd.Series(pd.ewma(abs(PosDI - NegDI) / (PosDI + NegDI), span=n_ADX, min_periods=n_ADX - 1),
                    name='ADX_' + str(n) + '_' + str(n_ADX))
    df = df.join(ADX)
    return df




# Mass Index
def get_mass_i(df):
    Range = df['High'] - df['Low']
    EX1 = pd.ewma(Range, span=9, min_periods=8)
    EX2 = pd.ewma(EX1, span=9, min_periods=8)
    Mass = EX1 / EX2
    MassI = pd.Series(pd.rolling_sum(Mass, 25), name='Mass Index')
    df = df.join(MassI)
    return df


# Vortex Indicator: http://www.vortexindicator.com/VFX_VORTEX.PDF
def get_vortex(df, n):
    i = 0
    TR = [0]
    while i < df.index[-1]:
        Range = max(df.get_value(i + 1, 'High'), df.get_value(i, 'Close')) - min(df.get_value(i + 1, 'Low'),
                                                                                 df.get_value(i, 'Close'))
        TR.append(Range)
        i = i + 1
    i = 0
    VM = [0]
    while i < df.index[-1]:
        Range = abs(df.get_value(i + 1, 'High') - df.get_value(i, 'Low')) - abs(
            df.get_value(i + 1, 'Low') - df.get_value(i, 'High'))
        VM.append(Range)
        i = i + 1
    VI = pd.Series(pd.rolling_sum(pd.Series(VM), n) / pd.rolling_sum(pd.Series(TR), n), name='Vortex_' + str(n))
    df = df.join(VI)
    return df


# KST Oscillator
def KST(df, r1, r2, r3, r4, n1, n2, n3, n4):
    M = df['Close'].diff(r1 - 1)
    N = df['Close'].shift(r1 - 1)
    ROC1 = M / N
    M = df['Close'].diff(r2 - 1)
    N = df['Close'].shift(r2 - 1)
    ROC2 = M / N
    M = df['Close'].diff(r3 - 1)
    N = df['Close'].shift(r3 - 1)
    ROC3 = M / N
    M = df['Close'].diff(r4 - 1)
    N = df['Close'].shift(r4 - 1)
    ROC4 = M / N
    KST = pd.Series(
        pd.rolling_sum(ROC1, n1) + pd.rolling_sum(ROC2, n2) * 2 + pd.rolling_sum(ROC3, n3) * 3 + pd.rolling_sum(ROC4,
                                                                                                                n4) * 4,
        name='KST_' + str(r1) + '_' + str(r2) + '_' + str(r3) + '_' + str(r4) + '_' + str(n1) + '_' + str(
            n2) + '_' + str(n3) + '_' + str(n4))
    df = df.join(KST)
    return df


# Relative Strength Index
def RSI(df, n):
    i = 0
    UpI = [0]
    DoI = [0]
    while i + 1 <= df.index[-1]:
        UpMove = df.get_value(i + 1, 'High') - df.get_value(i, 'High')
        DoMove = df.get_value(i, 'Low') - df.get_value(i + 1, 'Low')
        if UpMove > DoMove and UpMove > 0:
            UpD = UpMove
        else:
            UpD = 0
        UpI.append(UpD)
        if DoMove > UpMove and DoMove > 0:
            DoD = DoMove
        else:
            DoD = 0
        DoI.append(DoD)
        i = i + 1
    UpI = pd.Series(UpI)
    DoI = pd.Series(DoI)
    PosDI = pd.Series(pd.ewma(UpI, span=n, min_periods=n - 1))
    NegDI = pd.Series(pd.ewma(DoI, span=n, min_periods=n - 1))
    RSI = pd.Series(PosDI / (PosDI + NegDI), name='RSI_' + str(n))
    df = df.join(RSI)
    return df


# True Strength Index
def get_tsi(df, r, s):
    M = pd.Series(df['Close'].diff(1))
    aM = abs(M)
    EMA1 = pd.Series(pd.ewma(M, span=r, min_periods=r - 1))
    aEMA1 = pd.Series(pd.ewma(aM, span=r, min_periods=r - 1))
    EMA2 = pd.Series(pd.ewma(EMA1, span=s, min_periods=s - 1))
    aEMA2 = pd.Series(pd.ewma(aEMA1, span=s, min_periods=s - 1))
    TSI = pd.Series(EMA2 / aEMA2, name='TSI_' + str(r) + '_' + str(s))
    df = df.join(TSI)
    return df


# Accumulation/Distribution
def get_accdist(df, n):
    ad = (2 * df['Close'] - df['High'] - df['Low']) / (df['High'] - df['Low']) * df['Volume']
    M = ad.diff(n - 1)
    N = ad.shift(n - 1)
    ROC = M / N
    AD = pd.Series(ROC, name='Acc/Dist_ROC_' + str(n))
    df = df.join(AD)
    return df


# Chaikin Oscillator
def get_chaikin_osci(df):
    ad = (2 * df['Close'] - df['High'] - df['Low']) / (df['High'] - df['Low']) * df['Volume']
    Chaikin = pd.Series(pd.ewma(ad, span=3, min_periods=2) - pd.ewma(ad, span=10, min_periods=9), name='Chaikin')
    df = df.join(Chaikin)
    return df


# Money Flow Index and Ratio
def get_mfi(df, n):
    PP = (df['High'] + df['Low'] + df['Close']) / 3
    i = 0
    PosMF = [0]
    while i < df.index[-1]:
        if PP[i + 1] > PP[i]:
            PosMF.append(PP[i + 1] * df.get_value(i + 1, 'Volume'))
        else:
            PosMF.append(0)
        i = i + 1
    PosMF = pd.Series(PosMF)
    TotMF = PP * df['Volume']
    MFR = pd.Series(PosMF / TotMF)
    MFI = pd.Series(pd.rolling_mean(MFR, n), name='MFI_' + str(n))
    df = df.join(MFI)
    return df


# On-balance Volume
def get_obv(df, n):
    i = 0
    OBV = [0]
    while i < df.index[-1]:
        if df.get_value(i + 1, 'Close') - df.get_value(i, 'Close') > 0:
            OBV.append(df.get_value(i + 1, 'Volume'))
        if df.get_value(i + 1, 'Close') - df.get_value(i, 'Close') == 0:
            OBV.append(0)
        if df.get_value(i + 1, 'Close') - df.get_value(i, 'Close') < 0:
            OBV.append(-df.get_value(i + 1, 'Volume'))
        i = i + 1
    OBV = pd.Series(OBV)
    OBV_ma = pd.Series(pd.rolling_mean(OBV, n), name='OBV_' + str(n))
    df = df.join(OBV_ma)
    return df


# Force Index
def get_force(df, n):
    F = pd.Series(df['Close'].diff(n) * df['Volume'].diff(n), name='Force_' + str(n))
    df = df.join(F)
    return df


# Ease of Movement
def get_eom(df, n):
    EoM = (df['High'].diff(1) + df['Low'].diff(1)) * (df['High'] - df['Low']) / (2 * df['Volume'])
    Eom_ma = pd.Series(pd.rolling_mean(EoM, n), name='EoM_' + str(n))
    df = df.join(Eom_ma)
    return df


# Commodity Channel Index
def CCI(df, n):
    PP = (df['High'] + df['Low'] + df['Close']) / 3
    CCI = pd.Series((PP - pd.rolling_mean(PP, n)) / pd.rolling_std(PP, n), name='CCI_' + str(n))
    df = df.join(CCI)
    return df


# Coppock Curve
def get_copp(df, n):
    M = df['Close'].diff(int(n * 11 / 10) - 1)
    N = df['Close'].shift(int(n * 11 / 10) - 1)
    ROC1 = M / N
    M = df['Close'].diff(int(n * 14 / 10) - 1)
    N = df['Close'].shift(int(n * 14 / 10) - 1)
    ROC2 = M / N
    Copp = pd.Series(pd.ewma(ROC1 + ROC2, span=n, min_periods=n), name='Copp_' + str(n))
    df = df.join(Copp)
    return df


# Keltner Channel
def get_kelch(df, n):
    KelChM = pd.Series(pd.rolling_mean((df['High'] + df['Low'] + df['Close']) / 3, n), name='KelChM_' + str(n))
    KelChU = pd.Series(pd.rolling_mean((4 * df['High'] - 2 * df['Low'] + df['Close']) / 3, n), name='KelChU_' + str(n))
    KelChD = pd.Series(pd.rolling_mean((-2 * df['High'] + 4 * df['Low'] + df['Close']) / 3, n), name='KelChD_' + str(n))
    df = df.join(KelChM)
    df = df.join(KelChU)
    df = df.join(KelChD)
    return df


# Ultimate Oscillator
def get_ultosc(df):
    i = 0
    TR_l = [0]
    BP_l = [0]
    while i < df.index[-1]:
        TR = max(df.get_value(i + 1, 'High'), df.get_value(i, 'Close')) - min(df.get_value(i + 1, 'Low'),
                                                                              df.get_value(i, 'Close'))
        TR_l.append(TR)
        BP = df.get_value(i + 1, 'Close') - min(df.get_value(i + 1, 'Low'), df.get_value(i, 'Close'))
        BP_l.append(BP)
        i = i + 1
    UltO = pd.Series((4 * pd.rolling_sum(pd.Series(BP_l), 7) / pd.rolling_sum(pd.Series(TR_l), 7)) + (
    2 * pd.rolling_sum(pd.Series(BP_l), 14) / pd.rolling_sum(pd.Series(TR_l), 14)) + (
                     pd.rolling_sum(pd.Series(BP_l), 28) / pd.rolling_sum(pd.Series(TR_l), 28)), name='Ultimate_Osc')
    df = df.join(UltO)
    return df


# Donchian Channel
def get_donchian(df, n):
    i = 0
    DC_l = []
    while i < n - 1:
        DC_l.append(0)
        i = i + 1
    i = 0
    while i + n - 1 < df.index[-1]:
        DC = max(df['High'].ix[i:i + n - 1]) - min(df['Low'].ix[i:i + n - 1])
        DC_l.append(DC)
        i = i + 1
    don_ch = pd.Series(DC_l, name='Donchian_' + str(n))
    don_ch = don_ch.shift(n - 1)
    df = df.join(don_ch)
    return df

