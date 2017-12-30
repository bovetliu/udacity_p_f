import numpy as np
from hmmlearn import hmm
import warnings
from sklearn import neighbors, preprocessing
import pandas as pd
import seaborn as sns
import pprint
from analytic import utility, performance, ta_indicators, hmm_strategy, math_formula

import math
from matplotlib import cm
from matplotlib import pyplot as plt
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts

# np.set_printoptions(suppress=True)
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

    momentum_rescaled = utility.rescale(ta_indicators.get_rocp(aapl_series, 20))

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
                                                 interested_col=['Date', 'Open', 'Close', 'Volume'],
                                                 join_spy_for_data_integrity=False,
                                                 keep_spy_if_not_having_spy=False,
                                                 base_dir='/home/boweiliu/workrepo/udacity_p_f/src/rawdata')
    # reverse and select data required for study
    sha_001_df = sha_001_df.iloc[::-1].loc['2000-01-04':'2016-09-02']
    close = sha_001_df['SHA01_CLOSE']
    volume = sha_001_df['SHA01_VOLUME']
    date_list = pd.to_datetime(close.index)

    rtn = ta_indicators.get_daily_return(close)
    ln_rtn = ta_indicators.get_ln_return(close)
    assert len(ln_rtn) > 0
    rtn.iloc[0] = 0.001
    ln_rtn.iloc[0] = 0.00001
    print('len(sha_001_df_drtn) : {}, the total records number.'.format(len(ln_rtn)))
    split_idx = int(len(ln_rtn) / 5.0 * 4.0)
    print("split_idx: {}".format(split_idx))

    close_pds_train = close.iloc[0:split_idx]
    close_pds_test = close.iloc[split_idx:]

    volume_pds_train = volume.iloc[0:split_idx]
    volume_pds_test = volume.iloc[split_idx:]

    # normal return
    rtn_pds_train = rtn.iloc[0:split_idx]
    rtn_pds_test = rtn.iloc[split_idx:]

    # natural log return
    ln_rtn_train = ln_rtn.iloc[0:split_idx]
    ln_rtn_test = ln_rtn.iloc[split_idx:]

    date_list_train = date_list[0:split_idx]
    date_list_test = date_list[split_idx:]

    X = np.column_stack([np.array(ln_rtn_train)])
    X_test = np.column_stack([np.array(ln_rtn_test)])

    trained_hmm = hmm.GaussianHMM(n_components=6, covariance_type='full', n_iter=500)
    # only use sample section to train model
    trained_hmm.fit(X)
    # Predict the optimal sequence of internal hidden state
    hs_seq_train = trained_hmm.predict(X)
    hs_seq_test = trained_hmm.predict(X_test)

    print('len(hidden_states_seq_train) : {}'.format(len(hs_seq_train)))
    print('len(hidden_states_seq_test) : {}'.format(len(hs_seq_test)))

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
        state = (hs_seq_train == i)
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
    hs_class = []

    plt.figure(figsize=(15, 8))
    for state_i in range(trained_hmm.n_components):
        mask = [1 if state else 0 for state in (hs_seq_train == state_i)]
        mask = np.append([0], mask[:-1])  # right shift mask, which represents can only BUY/SELL at 2nd trading day
        log_cum_rtn = ln_rtn_train.multiply(mask, axis=0).cumsum()
        if log_cum_rtn.iloc[-1] > 0.5:
            # market up
            hs_class.append(1)
        elif log_cum_rtn.iloc[-1] < -0.5:
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

    # start cell 3
    # remapping hidden states sequence into assumed states sequence
    assumed_states_seq_train = np.array(
        [hs_class[hidden_state_i]
         for hidden_state_i in hs_seq_train])
    # -1 means market down
    # 0 means market vibrate
    # 1 means market up
    # print(assumed_states_seq_train == 2)

    plt.figure(figsize=(15, 8))
    for state_i in [-1, 0, 1]:
        mask = [1 if assume_state == state_i else 0 for assume_state in assumed_states_seq_train]
        mask = np.append([0], mask[:-1])  # right shift mask, which represents can only BUY/SELL at 2nd trading day
        assumed_states_rtn = ln_rtn_train.multiply(mask, axis=0)
        assumed_states_cum_rtn = assumed_states_rtn.cumsum()
        the_label = 'log-up' if state_i == 1 else 'log-down' if state_i == -1 else 'log-vibration'
        plt.plot(assumed_states_cum_rtn, label=the_label)
        plt.legend()
        plt.grid(1)
    # plt.show()

    # start cell 4
    # remapping hidden states sequence into assumed states sequence
    assumed_states_seq_test = np.array([hs_class[hsi] for hsi in hs_seq_test])
    # 0 means xia die
    # 1 means zhengdang
    # 2 means shangzhang
    # print(assumed_states_seq_train == 2)

    plt.figure(figsize=(15, 8))
    for state_i in [-1, 0, 1]:
        mask = [1 if assume_state == state_i else 0 for assume_state in assumed_states_seq_test]
        assumed_states_rtn_test = ln_rtn_test.multiply(mask, axis=0)
        assumed_states_cum_rtn_test = assumed_states_rtn_test.cumsum()
        the_label = 'log-up-test' if state_i == 1 else \
            'log-down-test' if state_i == -1 else 'log-vibration-test'
        plt.plot(assumed_states_cum_rtn_test, label=the_label)
        plt.legend()
        plt.grid(1)

    in_sample_df = sha_001_df.iloc[0: split_idx]
    test_df = sha_001_df.iloc[split_idx:]

    print("assumed_states_seq_train : {}".format(assumed_states_seq_train))
    in_sample_df = in_sample_df.assign(SHA01_SIGNAL=assumed_states_seq_train)
    test_df = test_df.assign(SHA01_SIGNAL=assumed_states_seq_test)

    net_worths_in_sample = performance.get_relative_net_worth(in_sample_df, 'SHA01')
    print(net_worths_in_sample)
    net_worths_test = performance.get_relative_net_worth(test_df, 'SHA01')
    print(net_worths_test)

    sharpe_ratio_in_sample = performance.get_sharp_ratio(net_worths_in_sample)
    sharpe_ratio_test = performance.get_sharp_ratio(net_worths_test)
    print(sharpe_ratio_in_sample, sharpe_ratio_test)
    plt.show()


def practice_hmm03():
    """
    using rescaled_norm, rescaled_sma_slope, rescaled_rel_std
    :return:
    """
    symbol = "AAPL"
    csv_files = ["{}_20100104-20171013".format(symbol)]
    # get raw, whole scale data
    the_df = utility.get_cols_from_csv_names(csv_files,
                                             interested_col=['Date', 'Close', 'Volume', 'Adj Close'],
                                             join_spy_for_data_integrity=True,
                                             keep_spy_if_not_having_spy=False)
    adj_close_ser = the_df['{}_ADJ_CLOSE'.format(symbol)]
    window_size = 20
    rolling_norm = ta_indicators.get_window_normalized(adj_close_ser, window_size=window_size)

    rolling_mean = ta_indicators.get_rolling_mean(adj_close_ser, window_size=window_size)
    slope_rolling_mean = ta_indicators.get_daily_return(rolling_mean)
    rolling_std = ta_indicators.get_rolling_std(adj_close_ser, window_size=window_size)
    rel_rolling_std = rolling_std / rolling_mean
    rel_rolling_std.rename("REL_{}".format(rolling_std.name), inplace=True)

    rescaled_norm = utility.rescale(rolling_norm, False, "RESCALED_NORM")
    rescaled_sma_slope = utility.rescale(slope_rolling_mean, False, "RESCALED_SMA_SLOPE")
    rescaled_rel_std = utility.rescale(rel_rolling_std, False, "RESCALED_REL_STD")
    combined_df = rescaled_norm.to_frame().join(rescaled_sma_slope, how="outer").join(rescaled_rel_std, how="outer")
    print(combined_df)
    # utility.plot_data(combined_df)
    hmm_strategy.eval(feature_cols=[rescaled_norm, rescaled_sma_slope, rescaled_rel_std],
                      close_col=adj_close_ser,
                      num_state=6)


def practice_hmm04():
    """
    using rescaled ln return
    :return:
    """
    symbol = "QQQ"
    csv_files = ["{}_2003-01-06_2017-11-28".format(symbol)]
    # get raw, whole scale data
    the_df = utility.get_cols_from_csv_names(csv_files,
                                             interested_col=['Date', 'Close', 'Volume', 'Adj Close'],
                                             join_spy_for_data_integrity=False,
                                             keep_spy_if_not_having_spy=False)
    adj_close_ser = the_df['{}_ADJ_CLOSE'.format(symbol)]

    ln_rtn = ta_indicators.get_ln_return(adj_close_ser)
    ln_rtn.iloc[0] = 0.001
    ln_rtn = utility.rescale(ln_rtn, False)
    print(ln_rtn)
    hmm_strategy.eval(feature_cols=[ln_rtn],
                      close_col=adj_close_ser,
                      num_state=9)


def practice_uvxy_shunhao():
    symbol = "UVXY"
    csv_files = [utility.get_appropriate_file(symbol)]
    requested_col = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close']
    the_df = utility.get_cols_from_csv_names(csv_files,
                                             interested_col=requested_col,
                                             join_spy_for_data_integrity=False,
                                             keep_spy_if_not_having_spy=False)
    the_df = the_df.loc[the_df.index >= '2012-01-01']
    print(the_df.head())
    rm_shunhao = ta_indicators.remove_shunhao(the_df['{}_ADJ_CLOSE'.format(symbol)])
    ax = rm_shunhao.plot(title='RM_SHUNHAO', legend=True)
    u, s, l = ta_indicators.get_bbands(rm_shunhao, 50)
    u.plot(ax=ax, legend=True)
    s.plot(ax=ax, legend=True)
    l.plot(ax=ax, legend=True)
    plt.show(block=True)


def practice_component_decomposing():
    symbol = "AMD"
    csv_files = [utility.get_appropriate_file(symbol)]
    requested_col = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    the_df = utility.get_cols_from_csv_names(csv_files,
                                             interested_col=requested_col,
                                             join_spy_for_data_integrity=False,
                                             keep_spy_if_not_having_spy=False)
    the_df = the_df.loc[the_df.index >= '2017-01-06']
    print(the_df.head(5))
    intra_day_rtns, gap_rtns = ta_indicators.get_daily_return_2(the_df)
    normal_rtn = ta_indicators.get_daily_return(the_df['{}_CLOSE'.format(symbol)]).add(1.0, axis=0).cumprod(axis=0)
    normal_rtn.name = 'AMD_RTN'
    intra_day_rtns_name, gap_rtns_name = intra_day_rtns.name, gap_rtns.name
    print(intra_day_rtns.head(10))

    print('mean: {}, std: {}'.format(intra_day_rtns.mean(), intra_day_rtns.std()))
    # intra_day_rtns.hist()

    intra_day_rtns_add_1 = intra_day_rtns.add(1.0).cumprod(axis=0)
    intra_day_rtns_add_1.name = 'AMD_INTRADAY_RTN'
    print(intra_day_rtns_add_1.head(10))
    ax = intra_day_rtns_add_1.plot(title="intra_day_rtn_cumprod", legend=True, figsize=(14, 7))
    gap_rtns_add_1 = gap_rtns.add(1.0, axis=0).cumprod(axis=0)
    gap_rtns_add_1.name = 'AMD_GAP_RTN'
    gap_rtns_add_1.plot(ax=ax, legend=True)
    normal_rtn.plot(ax=ax, legend=True)

    # intra_day_rtns = intra_day_rtns.cumprod(axis=0)



    # TODO(Bowei) revisit them later
    # gap_rtns = gap_rtns.cumprod(axis=0)
    # intra_day_rtns.name = intra_day_rtns_name
    # gap_rtns.name = gap_rtns_name

    # gap_rtns_upper, gap_rtns_sma, gap_rtns_lower = ta_indicators.get_bbands(gap_rtns, 30)

    # plt.figure(figsize=(9, 6))
    # ax2 = gap_rtns.plot(title='GAP COMPONENT', legend=True)
    # ax2.set_xlabel('Date')
    # ax2.set_ylabel('Return')
    # gap_rtns_upper.plot(ax=ax2, legend=True)
    # gap_rtns_sma.plot(ax=ax2, legend=True)
    # gap_rtns_lower.plot(ax=ax2, legend=True)
    plt.show(block=True)


def practice_np():
    a = np.array([0, 1, 2])
    print(np.tile(a, [2]))
    print("=================")
    print(np.tile(a, [1, 2]))
    print("=================")
    print(np.tile(a, [2, 1, 2]))
    print("=================")
    print(np.tile(np.identity(2), (3, 1, 1)))

    x_array = np.array(list(range(-200, 200))) / 100.0
    y_array = math_formula.perceptron(x_array, k=24, x_0=-0.8, y_span=0.2, b=0.004)
    print(y_array)

    plt.plot(x_array, y_array * 100)
    plt.xlabel('x')
    plt.ylabel('y%')
    plt.show()


def practice_algo_trading_chap2():
    symbols = ('EWA','EWC')
    csv_files = [utility.get_appropriate_file(symbol) for symbol in symbols]
    requested_col = ['Date', 'Adj Close']
    the_df = utility.get_cols_from_csv_names(file_names=csv_files,
                                             interested_col=requested_col,
                                             join_spy_for_data_integrity=False,
                                             keep_spy_if_not_having_spy=False)

    the_df = the_df.loc[(the_df.index >= '2006-04-04') & (the_df.index <= '2012-04-09')]
    print(the_df.head())
    show_plot1 = False
    show_scatter_plot = False

    if show_plot1:
        plot1 = the_df.plot(legend=True, title="EWA, EWC share prices")
        plot1.set_xlabel('Date')
        plot1.set_ylabel('Share price $')
        plt.show()

    # EWA as x array
    ewa_ser = the_df['{}_ADJ_CLOSE'.format(symbols[0])]
    ewc_ser = the_df['{}_ADJ_CLOSE'.format(symbols[1])]
    if show_scatter_plot:
        plt.scatter(
            ewa_ser.values,
            ewc_ser.values,
            marker='.',
            alpha=0.5
        )
        plt.xlabel(ewa_ser.name)
        plt.ylabel(ewc_ser.name)
        plt.title('Scatter Plot of EWA versus EWC')
        plt.show(block=True)
    print('len(ewa_ser): {}, len(ewc_ser): {}'.format(len(ewa_ser), len(ewc_ser)))

    # Calculate optimal hedge ratio "beta"
    ols_model = sm.OLS(ewc_ser.values, ewa_ser.values)
    ols_results = ols_model.fit()
    print('ols_results.params: {}'.format(ols_results.params))
    print('ols_results.tvalues: {}'.format(ols_results.tvalues))
    print(ols_results.summary())




if __name__ == "__main__":
    practice_algo_trading_chap2()
