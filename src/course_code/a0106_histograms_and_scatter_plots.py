"""Plot a histogram"""

import pandas as pd
import matplotlib.pyplot as plt

from course_code.a0102_python_for_finance01 import get_data_from_file_names, plot_data
from course_code.a0104_statistical_analysis_of_time_series import compute_daily_returns


def instructional_code_05_06():

    # read data
    dates = pd.date_range('2009-01-01', '2012-12-31')
    file_names = ['SPY_20090102-20171017']
    df = get_data_from_file_names(file_names, dates)
    # plot_data(df)

    # Compute daily returns
    daily_returns = compute_daily_returns(df)
    plot_data(daily_returns, title='Daily returns', ylabel="Daily returns")
    daily_returns.plot.hist(daily_returns, bins=20)  # changed bins to 20

    # Get mean and standard deviation
    mean = daily_returns['SPY'].mean()
    print("mean={}".format(mean))
    std = daily_returns['SPY'].std()
    print("std={}".format(std))

    plt.axvline(mean, color='w', linestyle='dashed', linewidth=2)
    plt.axvline(std, color='r', linestyle='dashed', linewidth=2)
    plt.axvline(-std, color='r', linestyle='dashed', linewidth=2)
    plt.show()

    print("daily_returns.kurtosis(): ")
    print(daily_returns.kurtosis())


def instructional_code_08():
    pass

# later on, alpha, beta, and correlation is discussed


if __name__ == "__main__":
    instructional_code_05_06()
