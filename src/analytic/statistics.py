from scipy import stats
import numpy as np


def pvalue(sample: np.ndarray, *argv, **kwargs):
    if 'mean2' not in kwargs:
        raise ValueError('mean2 not supplied in kwargs')
    if 'std2' not in kwargs:
        raise ValueError('mean2 not supplied in kwargs')
    if 'nobs2' not in kwargs:
        raise ValueError('mean2 not supplied in kwargs')
    t, p = stats.ttest_ind_from_stats(
        sample.mean(),
        sample.std(),
        len(sample),
        kwargs['mean2'],
        kwargs['std2'],
        kwargs['nobs2'],
        equal_var=False)
    return p


def linear_regression(sample: np.ndarray, *argv, **kwargs):
    """
    calculate fitted rate of change per minute in sample, than divided by std of sample 2 (whole scale sample)
    """
    if 'mean2' not in kwargs:
        raise ValueError('mean2 not supplied in kwargs')
    if 'std2' not in kwargs:
        raise ValueError('mean2 not supplied in kwargs')
    if 'nobs2' not in kwargs:
        raise ValueError('mean2 not supplied in kwargs')
    sample_len = len(sample)
    if sample_len <= 1:
        return 0
    x = np.arange(0, sample_len)
    y = sample
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    # print("slope: {}, slope / std2: {}".format(slope, slope / kwargs['std2']))
    return slope / sample[0] / kwargs['std2']


def drop_down(sample: np.ndarray, *argv, **kwargs):
    if 'mean2' not in kwargs:
        raise ValueError('mean2 not supplied in kwargs')
    if 'std2' not in kwargs:
        raise ValueError('mean2 not supplied in kwargs')
    if 'nobs2' not in kwargs:
        raise ValueError('mean2 not supplied in kwargs')
    sample_len = len(sample)
    if sample_len < 2:
        return 0
    max_drop_down_rate = 0
    index_diff = 1
    for i in range(sample_len):
        for j in range(i + 1, sample_len):
            drop_down_rate = (sample[j] - sample[i]) / sample[i]
            if drop_down_rate < max_drop_down_rate:
                index_diff = j - i
                max_drop_down_rate = drop_down_rate
    return max_drop_down_rate / index_diff / kwargs['std2']

