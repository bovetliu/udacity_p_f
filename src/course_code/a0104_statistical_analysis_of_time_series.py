import pandas as pd
import matplotlib.pyplot as plt

from course_code import a0102_python_for_finance01


def instructional_code_04():
    dates = pd.date_range('2010-01-01', '2012-12-31')
    file_names = [
        'SPY_20100104-20171013',
        'XOM_20100104-20171013',
        'GOOG_20100104-20171013',
        'GLD_20100104-20171013'
    ]
    df = a0102_python_for_finance01.get_data_from_file_names(file_names, dates)
    # a0102_python_for_finance01.plot_data(df)

    # Compute global statistics for each stock
    print("mean")
    print(df.mean())

    print("\ndf.median()")
    print(df.median())

    print("\ndf.std()")
    print(df.std())


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


def instructional_code_08():
    dates = pd.date_range('2012-01-01', '2012-12-31')
    file_names = ['SPY_20100104-20171013']
    df = a0102_python_for_finance01.get_data_from_file_names(file_names, dates)

    # Plot SPY data, retain matplotlib axis object
    ax = df['SPY'].plot(title="SPY rolling mean", label='SPY')

    # Compute rolling mean using a 20-day window
    # new version of pandaspd
    rolling_mean_of_spy = df['SPY'].rolling(window=20).mean()
    rolling_mean_of_spy.plot(label='Rolling mean', ax=ax)

    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(loc="upper left")
    plt.show()


def compute_daily_returns(df: pd.DataFrame) -> pd.DataFrame:
    daily_return = df.copy()

    # numpy 1-D ndarray arithmic operation minus by 1
    daily_return[1:] = (df[1:].values / df[:-1].values) - 1

    # noinspection PyUnresolvedReferences
    daily_return.iloc[0, :] = 0  # set row 0, every column to 0
    # noinspection PyTypeChecker
    return daily_return


def instructional_code_11():
    dates = pd.date_range('2012-07-01', '2012-07-31')
    file_names = ['SPY_20100104-20171013', 'XOM_20100104-20171013']
    df = a0102_python_for_finance01.get_data_from_file_names(file_names, dates)
    a0102_python_for_finance01.plot_data(df)

    # Compute daily returns
    daily_returns = compute_daily_returns(df)
    a0102_python_for_finance01.plot_data(daily_returns, title="Daily returns")


if __name__ == "__main__":
    instructional_code_11()
