import numpy as np
from hmmlearn import hmm
import warnings
from sklearn import neighbors, preprocessing
import pandas as pd
import seaborn as sns
from analytic import utility
from analytic import ta_indicators

from matplotlib import cm
from matplotlib import pyplot as plt

np.set_printoptions(suppress=True)
warnings.filterwarnings("ignore", category=DeprecationWarning)


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
    csv_files = ['SHA01']
    sha_001_df = utility.get_cols_from_csv_names(csv_files, interested_col=['Date', 'Close', 'Volume'], join_spy=False)
    # reverse
    sha_001_df = sha_001_df.iloc[::-1].loc['2000-01-04':'2016-09-02']

    sha_001_df_drtn = ta_indicators.get_daily_return(sha_001_df['SHA01_CLOSE'])
    sha_001_df_drtn.iloc[0] = 0

    print('len(sha_001_df_drtn) : {}'.format(sha_001_df_drtn))
    sha_001_df_drtn_nparray = np.array(sha_001_df_drtn)
    X = np.column_stack([sha_001_df_drtn_nparray])

    # transformed = preprocessing.quantile_transform(X, output_distribution='normal', axis=0)
    # print(np.std(transformed, axis=0))
    # print(np.mean(transformed, axis=0))
    # input_matrix_normalized = preprocessing.normalize(input_matrix, norm='l2')
    # np.set_printoptions(threshold=5000, suppress=True)
    # print(np.column_stack([np.transpose(transformed)[0], sha_001_df_drtn_nparray]))
    trained_hmm = hmm.GaussianHMM(n_components=8, covariance_type='diag', n_iter=2000).fit(X)
    trained_hmm.transmat_ = np.round(trained_hmm.transmat_, 5)
    # Predict the optimal sequence of internal hidden state
    hidden_states_seq = trained_hmm.predict(X)
    print("trained_hmm.transmat_:")
    print(trained_hmm.transmat_)
    print('len(hidden_states_seq) : {}'.format(len(hidden_states_seq)))

    print("Means and vars of each hidden state")
    for i in range(trained_hmm.n_components):
        print("{0}th hidden state".format(i))
        print("mean = ", trained_hmm.means_[i])
        print("var = ", np.diag(trained_hmm.covars_[i]))
        print()

    sns.set_style('white')
    plt.figure(figsize=(15, 8))
    for i in range(trained_hmm.n_components):
        state = (hidden_states_seq == i)
        print(state)
        # plt.plot(datelist[state], closeidx[state], '.', label='latent state %d' % i, lw=1)
        # plt.legend()
        # plt.grid(1)

def practice_hmm03():
    pass



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
