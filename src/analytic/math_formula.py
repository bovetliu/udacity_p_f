import numpy as np
import math


def perceptron(x, k: float=1, x_0: float=0, y_span: float=1, b: float=0):
    """
    perceptron y calculation
    :param x: can be a numpy series or a single x
    :param k: how drastically the function change when (x - x0) is close to 0
    :param x_0: horizontal shift
    :param y_span: vertal span,
    :param b: b
    :return: y_span / (1 + exp(-k * (x - x_0))) + b
    """
    return y_span / (1 + np.exp(-k * (x - x_0))) + b
