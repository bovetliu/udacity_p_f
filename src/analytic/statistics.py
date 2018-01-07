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
