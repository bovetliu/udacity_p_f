import numpy as np
from hmmlearn import hmm
from sklearn import neighbors
# import matplotlib.pyplot as plt
import pandas as pd

from analytic import utility
from analytic import ta_indicators


def polarize(x):
    return 1 if x > 0 else (0 if x == 0 else -1)


def practice_knn():
    """
    practice KNN prediction
    """
    window_size = 20
    csv_files = ['AAPL_20100104-20171013']
    aapl_df = utility.get_cols_from_csv_names(csv_files)
    # utility.plot_data(aapl_df)
    aapl_series = aapl_df['AAPL']
    aapl_simple_mean = ta_indicators.get_rolling_mean(aapl_series, window_size)
    aapl_std = ta_indicators.get_rolling_std(aapl_series, window_size)

    rel_pos_bb = ta_indicators.get_window_normalized(aapl_series, window_size, 2)
    rescaled_volatility = utility.rescale(aapl_std / aapl_simple_mean)
    rescaled_volatility.name = 'AAPL_VOLATILITY'
    aapl_simple_mean_drtn_rescaled = utility.rescale(ta_indicators.get_daily_return(aapl_simple_mean))
    vol_window_normlzd = ta_indicators.get_window_normalized(aapl_df['AAPL_VOL'], window_size)

    momentum_rescaled = utility.rescale(ta_indicators.get_momentum(aapl_series, 20))

    # following are labels, without rescaling
    future_returns = ta_indicators.get_frws(aapl_series)
    # print(rel_pos_bb)

    # combined_df = pd.concat([rel_pos_bb,
    #                          rescaled_volatility,
    #                          aapl_simple_mean_drtn_rescaled,
    #                          vol_window_normlzd,
    #                          momentum_rescaled], axis=1)
    combined_df = pd.DataFrame(pd.concat([rel_pos_bb,
                                          rescaled_volatility,
                                          aapl_simple_mean_drtn_rescaled], axis=1))

    input_mx = np.array(combined_df.iloc[20: -1, :])
    output = np.array(future_returns.iloc[20: -1])
    output = np.array([polarize(x) for x in output])

    train_size = len(input_mx) - 252
    output_test = output[-252:]
    output_pred = []
    for i in range(train_size - 252, train_size, 1):
        n_neighbors = 3
        knn = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, weights='uniform')
        knn.fit(input_mx[i - 1008:i], output[i - 1008:i])

        # print(input_mx[i: i + 1])
        predict = knn.predict(input_mx[i: i + 1])
        output_pred.append(predict[0])

    cnt_correct = 0
    for i in range(len(output_pred)):
        if output_pred[i] == output_test[i]:
            cnt_correct += 1
    print(cnt_correct * 1.0 / len(output_pred))


def practice_hmm():
    np.random.seed(42)

    model = hmm.GaussianHMM(n_components=3, covariance_type='full')

    # at initial time, the probabilities distribution of internal state
    model.startprob_ = np.array([0.6, 0.3, 0.1])

    # transition matrix
    model.transmat_ = np.array([[0.7, 0.2, 0.1],
                                [0.3, 0.5, 0.2],
                                [0.3, 0.3, 0.4]])

    model.means_ = np.array([[0.0, 0.0],
                             [3.0, -3.0],
                             [5.0, 10.0]])

    model.covars_ = np.tile(np.identity(2), (3, 1, 1))
    X, Z = model.sample(100)
    print(X)  # sequence of observable X
    print(Z)  # a sequence of hidden state


def practice_hmm02():
    csv_files = ['SHA000001']
    sha_001_df = utility.get_cols_from_csv_names(csv_files, interested_col=['Date', 'Close', 'Volume'], join_spy=False)
    # reverse
    sha_001_df = sha_001_df.iloc[::-1]
    # according to page 9, model cross validation, whole data set
    sha_001_df = sha_001_df.loc['2000-01-04':'2016-09-02']
    print(sha_001_df)


def practice_np():
    a = np.array([0, 1, 2])
    print(np.tile(a, [2]))
    print("=================")
    print(np.tile(a, [1, 2]))
    print("=================")
    print(np.tile(a, [2, 1, 2]))
    print("=================")
    print(np.tile(np.identity(2), (3, 1, 1)))


if __name__ == "__main__":
    practice_hmm02()
