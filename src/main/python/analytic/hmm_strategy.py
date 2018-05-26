from typing import List

import numpy as np
from hmmlearn import hmm
import warnings
from sklearn import neighbors, preprocessing
import pandas as pd
import seaborn as sns
from analytic import utility, performance, ta_indicators

from matplotlib import cm
from matplotlib import pyplot as plt


def eval(feature_cols: List[pd.Series], close_col: pd.Series, num_state: int=6):
    """
    evaluate whether these features can be used as HMM inputs.
    :param feature_cols: feature columns
    :param close_col: close column
    :param num_state: number of hidden state
    :return:
    """
    if not isinstance(feature_cols, list):
        raise TypeError("feature_cols should be object of python list")
    if not isinstance(close_col, pd.Series):
        raise TypeError("close_col should be a str")
    if len(feature_cols) <= 0:
        raise ValueError("feature_cols should at least one feature column")
    df = feature_cols[0].to_frame()
    for feature_col in feature_cols[1:]:
        df = df.join(feature_col, how='outer')
    df = df.join(close_col, how='outer')
    original_len = len(df)
    df.dropna(inplace=True)
    print("len(df) : {}, {} rows were removed.".format(len(df), original_len - len(df)))
    split_idx = int(len(df) / 5.0 * 4.0)
    print("split_idx: {}".format(split_idx))

    features_train = [df[feature_col.name].iloc[0:split_idx] for feature_col in feature_cols]
    features_test = [df[feature_col.name].iloc[split_idx:] for feature_col in feature_cols]

    # close
    close_ser = df[close_col.name]
    close_ser_train = close_ser.iloc[0:split_idx]
    close_ser_test = close_ser.iloc[split_idx:]

    # returns
    ln_rtn = ta_indicators.get_ln_return(df[close_col.name])
    ln_rtn.iloc[0] = 0.0001
    ln_rtn_train = ln_rtn.iloc[0:split_idx]
    ln_rtn_test = ln_rtn.iloc[split_idx:]

    date_list = pd.to_datetime(ln_rtn.index)
    date_list_train = date_list[0:split_idx]
    date_list_test = date_list[split_idx:]

    # these are Xs
    train_input_mx = np.column_stack([np.array(f_ser) for f_ser in features_train])
    test_input_mx = np.column_stack([np.array(f_ser) for f_ser in features_test])

    print(train_input_mx)
    trained_hmm = hmm.GaussianHMM(n_components=num_state, covariance_type='diag', n_iter=500)
    trained_hmm.fit(train_input_mx)

    # Predict the optimal sequence of internal hidden state (hs)
    hs_seq_train = trained_hmm.predict(train_input_mx)
    hs_seq_test = trained_hmm.predict(test_input_mx)
    print('len(hs_seq_train) : {}'.format(len(hs_seq_train)))
    print('len(hs_seq_test) : {}'.format(len(hs_seq_test)))
    print("\ntrained_hmm.transmat_:")
    print(trained_hmm.transmat_)
    print("\nMeans and vars of each hidden state")
    for feature_col in range(trained_hmm.n_components):
        print("{0}th hidden state".format(feature_col))
        print("mean = ", trained_hmm.means_[feature_col])
        print("var = ", np.diag(trained_hmm.covars_[feature_col]))
        print()

    sns.set_style('white')

    # trading day close marked by hidden state [TRAIN]
    plt.figure(figsize=(7, 4))
    for feature_col in range(trained_hmm.n_components):
        state = (hs_seq_train == feature_col)
        #  https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html#matplotlib.pyplot.plot
        plt.plot(date_list_train[state],  # x series
                 close_ser_train.iloc[state],  # y series
                 '.',  # point marker
                 label='hidden state {}'.format(feature_col),
                 linewidth=1)
        plt.legend()
        plt.grid(1)
    # plt.show()

    # start cell 2 ln return corresponding hidden state [TRAIN]
    hs_class = []
    plt.figure(figsize=(7, 4))
    for state_i in range(trained_hmm.n_components):
        mask = [1 if state else 0 for state in (hs_seq_train == state_i)]
        mask = np.append([0], mask[:-1])  # right shift mask, which represents prediction
        log_cum_rtn = ln_rtn_train.multiply(mask, axis=0).cumsum()
        if log_cum_rtn.iloc[-1] > 0.25:
            # market up
            hs_class.append(1)
        elif log_cum_rtn.iloc[-1] < -0.25:
            # market down
            hs_class.append(-1)
        else:
            # market vibrate
            hs_class.append(0)
        plt.plot(log_cum_rtn, label='hidden state {}'.format(state_i))
        plt.legend()
        plt.grid(1)
    print("hidden_state_classification : \n {}".format(hs_class))
    # plt.show()

    # start cell 3 ln return plot using labeled state [TRAIN]
    # remapping hidden states sequence into assumed states sequence
    assumed_states_seq_train = np.array([hs_class[hsi] for hsi in hs_seq_train])
    # 1  means market up
    # 0  means market vibrate
    # -1 means market down

    plt.figure(figsize=(7, 4))
    for state_i in [-1, 0, 1]:
        mask = [1 if assume_state == state_i else 0 for assume_state in assumed_states_seq_train]
        mask = np.append([0], mask[:-1])  # right shift mask, which represents being only able to BUY/SELL at 2nd day
        assumed_states_rtn = ln_rtn_train.multiply(mask, axis=0)
        assumed_states_cum_rtn = assumed_states_rtn.cumsum()
        the_label = 'ln-up-train' if state_i == 1 else 'ln-down-train' if state_i == -1 else 'ln-vibration-train'
        plt.plot(assumed_states_cum_rtn, label=the_label)
        plt.legend()
        plt.grid(1)
    # plt.show()

    # start cell 4 -trading day close marked by hidden state [TEST]
    plt.figure(figsize=(7, 4))
    for feature_col in range(trained_hmm.n_components):
        state = (hs_seq_test == feature_col)
        #  https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html#matplotlib.pyplot.plot
        plt.plot(date_list_test[state],  # x series
                 close_ser_test.iloc[state],  # y series
                 '.',  # point marker
                 label='hidden state {}'.format(feature_col),
                 linewidth=1)
        plt.legend()
        plt.grid(1)
    # plt.show()

    # start cell 5 ln return corresponding hidden state [TEST]
    plt.figure(figsize=(7, 4))
    for state_i in range(trained_hmm.n_components):
        mask = [1 if state else 0 for state in (hs_seq_test == state_i)]
        mask = np.append([0], mask[:-1])  # right shift mask, which represents being only able to BUY/SELL at 2nd day
        log_cum_rtn = ln_rtn_test.multiply(mask, axis=0).cumsum()
        plt.plot(log_cum_rtn, label='hidden state {}'.format(state_i))
        plt.legend()
        plt.grid(1)

    # start cell 6 ln return plot using labeled state [TEST]
    assumed_states_seq_test = np.array([hs_class[hsi] for hsi in hs_seq_test])
    # 1  means market up
    # 0  means market vibrate
    # -1 means market down

    plt.figure(figsize=(7, 4))
    for state_i in [-1, 0, 1]:
        mask = [1 if assume_state == state_i else 0 for assume_state in assumed_states_seq_test]
        mask = np.append([0], mask[:-1])  # right shift mask, which represents can only BUY/SELL at 2nd trading day
        assumed_states_rtn_test = ln_rtn_test.multiply(mask, axis=0)
        assumed_states_cum_rtn_test = assumed_states_rtn_test.cumsum()
        the_label = 'ln-up-test' if state_i == 1 else 'ln-down-test' if state_i == -1 else 'ln-vibration-test'
        plt.plot(assumed_states_cum_rtn_test, label=the_label)
        plt.legend()
        plt.grid(1)

    plt.show()
