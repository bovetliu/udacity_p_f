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
    aapl_df = utility.get_cols_from_csv_names(csv_files, keep_spy_if_not_having_spy=False)
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
    # start cell 1
    csv_files = ['SHA01']

    # get raw, whole scale data
    sha_001_df = utility.get_cols_from_csv_names(csv_files,
                                                 interested_col=['Date', 'Close', 'Volume'],
                                                 keep_spy_if_not_having_spy=False,
                                                 base_dir='/home/boweiliu/workrepo/udacity_p_f/src/rawdata')
    # reverse and select data required for study
    sha_001_df = sha_001_df.iloc[::-1].loc['2000-01-04':'2016-09-02']

    close = sha_001_df['SHA01_CLOSE']
    volume = sha_001_df['SHA01_VOLUME']
    date_list = pd.to_datetime(close.index)

    original_rtn = ta_indicators.get_daily_return(sha_001_df['SHA01_CLOSE'])
    ln_rtn = ta_indicators.get_ln_return(close)
    ln_rtn.iloc[0] = 0.00001
    print('len(sha_001_df_drtn) : {}, the total records number.'.format(len(ln_rtn)))
    split_idx = int(len(ln_rtn) / 5.0 * 4.0)
    print("split_idx: {}".format(split_idx))

    drtn_nparray_train = np.array(ln_rtn.iloc[0: split_idx])
    drtn_nparray_test = np.array(ln_rtn.iloc[split_idx:])

    close_pds_train = close.iloc[0:split_idx]
    close_pds_test = close.iloc[split_idx:]

    volume_pds_train = volume.iloc[0:split_idx]
    volume_pds_test = volume.iloc[split_idx:]

    ln_rtn_pds_train = ln_rtn.iloc[0:split_idx]
    ln_rtn_pds_test = ln_rtn.iloc[split_idx:]

    date_list_train = date_list[0:split_idx]
    date_list_test = date_list[split_idx:]

    X = np.column_stack([drtn_nparray_train])
    X_test = np.column_stack([drtn_nparray_test])

    trained_hmm = hmm.GaussianHMM(n_components=6, covariance_type='diag', n_iter=50)
    # only use sample section to train model
    trained_hmm.fit(X)
    # Predict the optimal sequence of internal hidden state
    hidden_states_seq_train = trained_hmm.predict(X)
    h_s_seq_test = trained_hmm.predict(X_test)

    print('len(hidden_states_seq_train) : {}'.format(len(hidden_states_seq_train)))
    print('len(hidden_states_seq_test) : {}'.format(len(h_s_seq_test)))

    print("\ntrained_hmm.transmat_:")
    print(trained_hmm.transmat_)
    print("\nMeans and vars of each hidden state")
    for i in range(trained_hmm.n_components):
        print("{0}th hidden state".format(i))
        print("mean = ", trained_hmm.means_[i])
        print("var = ", np.diag(trained_hmm.covars_[i]))
        print()

    sns.set_style('white')
    plt.figure(figsize=(15, 8))
    for i in range(trained_hmm.n_components):
        state = (hidden_states_seq_train == i)
        #  https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html#matplotlib.pyplot.plot
        plt.plot(date_list_train[state],  # x series
                 close_pds_train.iloc[state],  # y series
                 '.',  # point marker
                 label='hidden state {}'.format(i),
                 linewidth=1)
        plt.legend()
        plt.grid(1)
    # plt.show()

    # start cell 2
    h_s_class = []

    plt.figure(figsize=(15, 8))
    for state_i in range(trained_hmm.n_components):
        mask = [1 if state else 0 for state in (hidden_states_seq_train == state_i)]
        log_cum_rtn = ln_rtn_pds_train.multiply(mask, axis=0).cumsum()
        if log_cum_rtn.iloc[-1] > 0.5:
            # shang zhang hangqing
            h_s_class.append(2)
        elif log_cum_rtn.iloc[-1] < -0.5:
            # xia die hang qing
            h_s_class.append(0)
        else:
            # zheng dang hang qing
            h_s_class.append(1)
        plt.plot(log_cum_rtn, label='hidden state {}'.format(state_i))
        plt.legend()
        plt.grid(1)
    print("hidden_state_classification : \n {}".format(h_s_class))
    # plt.show()

    # start cell 3
    # remapping hidden states sequence into assumed states sequence
    assumed_states_seq_train = np.array(
        [h_s_class[hidden_state_i]
         for hidden_state_i in hidden_states_seq_train])
    # 0 means xia die
    # 1 means zhengdang
    # 2 means shangzhang
    # print(assumed_states_seq_train == 2)

    plt.figure(figsize=(15, 8))
    for state_i in range(3):
        mask = [1 if assume_state == state_i else 0 for assume_state in assumed_states_seq_train]
        assumed_states_rtn = ln_rtn_pds_train.multiply(mask, axis=0)
        assumed_states_cum_rtn = assumed_states_rtn.cumsum()
        the_label = 'log-up' if state_i == 2 else 'log-down' if state_i == 0 else 'log-vibration'
        plt.plot(assumed_states_cum_rtn, label=the_label)
        plt.legend()
        plt.grid(1)
    # plt.show()

    # start cell 4
    # remapping hidden states sequence into assumed states sequence
    assumed_states_seq_test = np.array([h_s_class[hsi] for hsi in h_s_seq_test])
    # 0 means xia die
    # 1 means zhengdang
    # 2 means shangzhang
    # print(assumed_states_seq_train == 2)

    plt.figure(figsize=(15, 8))
    for state_i in range(3):
        mask = [1 if assume_state == state_i else 0 for assume_state in assumed_states_seq_test]
        assumed_states_rtn_test = ln_rtn_pds_test.multiply(mask, axis=0)
        assumed_states_cum_rtn_test = assumed_states_rtn_test.cumsum()
        the_label = 'log-up-test' if state_i == 2 else \
            'log-down-test' if state_i == 0 else 'log-vibration-test'
        plt.plot(assumed_states_cum_rtn_test, label=the_label)
        plt.legend()
        plt.grid(1)
    plt.show()


def practice_np():
    a = np.array([0, 1, 2])
    print(np.tile(a, [2]))
    print("=================")
    print(np.tile(a, [1, 2]))
    print("=================")
    print(np.tile(a, [2, 1, 2]))
    print("=================")
    print(np.tile(np.identity(2), (3, 1, 1)))
    data_frame = utility.get_cols_from_csv_names(['AAPL_20100104-20171013'],
                                                 keep_spy_if_not_having_spy=False,
                                                 interested_col=['Date', 'Close', 'Open', 'Volume'])

    data_frame = data_frame.loc['2017-09-28':'2017-10-13']
    signal = [0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0]
    data_frame = data_frame.assign(AAPL_SIGNAL=pd.Series(signal).values)

    net_worth_ser = utility.get_relative_net_worth(data_frame, 'AAPL')
    print(net_worth_ser)



if __name__ == "__main__":
    practice_np()
