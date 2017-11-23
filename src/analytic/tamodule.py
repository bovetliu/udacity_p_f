import pandas as pd


def get_rolling_ema(values, window):
    """return EMA, compared with Yahoo finance"""
    return pd.Series(values).ewm(span=window, adjust=False).mean()


def get_rolling_mean(values, window):
    """Return rolling mean of given values, using specified window size."""
    return pd.Series(values).rolling(window=window).mean()


def get_rolling_std(values, window):
    """Return rolling standard deviation of given values, using specified window size."""
    return pd.Series(values).rolling(window=window).std()


def get_bollinger_bands(rolling_mean, rolling_std):
    """Return upper and lower Bollinger Bands."""
    upper_band = rolling_mean + 2 * rolling_std
    lower_band = rolling_mean - 2 * rolling_std
    return upper_band, lower_band
