"""
How to use an optimizer
1) Provide a function to minimize
  f(x) = x^2 + 0.5
2) Provide an initial guess
  x = 3
3) call the optimizer


Less name : Minimize an objective function, using SciPy
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spo


def f(X):
    """Given a scalar X, return some value (a real number)"""
    Y = (X - 1.5)**2 + 0.5
    print("X = {}, Y = {}".format(X, Y))
    return Y


def error(line, data):
    return np.sum((data[:, 1] - (line[0] * data[:, 0] + line[1])) ** 2)


def fit_line(data, error_func):
    # Generate initial guess for line model
    l = np.array([0, np.mean(data[:, 1])], np.float32)

    # Plot initial guess(optional)
    x_ends = np.array([-5, -3, 0, 1, 5, 10, 15], dtype=np.float32)
    plt.plot(x_ends, l[0] * x_ends + l[1], 'm--', linewidth=2.0, label="Initial Guess")

    # Call optimizer to minimize error function
    result = spo.minimize(error_func, l, args=(data,), method='SLSQP', options={'disp': True})
    return result.x


def instructional_code03():
    x_guess = np.array([2.0], dtype=np.float32)
    min_result = spo.minimize(f, x_guess, method="SLSQP", options={'disp': True})
    print("Minimal found at:")
    print("X = {}, Y = {}".format(min_result.x, min_result.fun))

    # Plot function values, mar minimal
    x_values = np.linspace(0.5, 2.5, 21)
    y_values = f(x_values)

    plt.plot(x_values, y_values)
    plt.plot(min_result.x, min_result.fun, 'ro')
    plt.title("Minimal of an objective function")
    plt.show()


def instructional_code09():
    l_origin = np.float32([4, 2])
    print("Original line: C0 = {}, C1 = {}".format(l_origin[0], l_origin[1]))
    x_origin_values = np.linspace(0, 10, 21)
    y_origin_values = l_origin[0] * x_origin_values + l_origin[1]
    origin_line_handle = plt.plot(x_origin_values, y_origin_values, 'b--', linewidth=2.0, label='Original line')
    # plt.legend(handles=origin_line_handle, loc=1)

    # Generate noisy data points
    noise_sigma = 3.0
    noise = np.random.normal(0, noise_sigma, y_origin_values.shape)
    # print(np.asarray([x_origin_values, y_origin_values + noise]).T)
    data = np.asarray([x_origin_values, y_origin_values + noise]).T
    data_points_handle = plt.plot(data[:, 0], data[:, 1], 'go', label="Data Points")
    # plt.legend(handles=data_points_handle, loc=1)
    l_fit = fit_line(data, error)
    print("Fitted line: C0 = {}, C1 = {}".format(l_fit[0], l_fit[1]))
    fitted_line_handle = plt.plot(data[:, 0], l_fit[0] * data[:, 0] + l_fit[1], 'r--', linewidth=2.0, label='fit line')
    plt.legend(handles=[origin_line_handle[0], fitted_line_handle[0], data_points_handle[0]], loc=1)
    plt.show()


if __name__ == "__main__":
    instructional_code09()
