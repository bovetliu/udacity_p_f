import numpy as np
import time

# row index and col index start from 0
# nd1[row, col]

# nd1[0:3. 1:3] # row 0,1,2 and col 1,2

# nd1[:, 3] # all rows, and col 3

# the last row -1, the last col -1
# nd1[-1, 1:3]


def instructional_code():
    # creating 1D array
    print(np.array([2, 3, 4]))

    # creating 2D array
    print(np.array([
        [2, 3, 4],
        [5, 6, 7]]
    ))

    # emtpy array  01-03 7.
    print(np.empty(4))
    print(np.empty((5, 4)))

    # one np array of 5 * 4, populated with 1s, and specify data type
    print(np.ones((5, 4), dtype=np.int_))

    # Gnenrating random numbers 01-03 9.
    # generate an array full of random numbers, uniformly sampled from [0.0, 1.0)
    print(np.random.random((5, 4)))

    # similar syntax
    # np.random.rand(5, 4)

    # normal distribution
    print(np.random.normal(loc=0.0, scale=1.0, size=(2, 3)))

    # random integers
    print(np.random.randint(low=0, high=10, size=(2, 3), dtype=np.int8))


def instructional_code_02():
    a = np.random.random(size=(5, 4))
    print(a)
    print(a.shape)
    print(a.size)
    print(a.dtype)

    # Operations on ndarrays 01-03
    np.random.seed(693)
    a = np.random.randint(low=0, high=10, size=(5, 4))
    print("\n\nArray:\n", a)

    print("Sum of all elements: {}".format(a.sum()))

    # Iterate over rows, to compute sum of each column
    print("Sum of each column:\n", a.sum(axis=0))

    # Iterate over columns to compute sum of each row
    print("Sum of each row: \n", a.sum(axis=1))

    # Statistics: min, max, mean (access rows, cols, and overall)
    print("Minumum of each column:\n", a.min(axis=0))
    print("Maximum of each row:\n", a.max(axis=1))
    print("Mean of all elements:\n", a.mean())  # leave out axis arg


def get_max_index(a:np.ndarray) -> int:
    """

    :param a: one ndarray
    :return: index of maximum element of a one-dimension ndarray
    """
    return a.argmax(axis=0)


def instructional_code_12():
    a = np.array([9, 6, 2, 3, 12, 14, 7, 10], dtype=np.int32)
    print("Array: ", a)
    print("Maximum value: ", a.max())
    print("Index of max.:", get_max_index(a))


def instructional_code_13():
    """
    Timing python operation
    :return: None
    """
    t1 = time.time()
    print("ML4T")
    t2 = time.time()
    print("The time take by print statement is {} seconds".format(t2 - t1))


def instructional_code_15_16():
    a = np.random.rand(5, 4)
    print("Array :\n", a)

    # Accessing element at position (3, 2)
    element = a[3, 2]
    print(element)

    # Elements in defined range
    print(a[0, 1:3])

    # left top corner
    print(a[0:2, 0:2])

    # Not : slice n:m:t specifies a range that starts at n, and stops before m, increment is t
    print(a[:, 0:3:2])

    print()
    # 16. Modifying array element
    # a[0, 0] = 1  # assign one cel
    # a[0, :] = 2  # assign entire row
    a[:, 3] = [1, 2, 3, 4, 5]
    print(a)


def instructional_code_17_18():
    a = np.random.rand(5)
    indices = np.array([1, 1, 2, 3])
    # print(a[indices])

    a = np.array([
        (20, 25, 10, 23, 26, 32, 10, 6, 0),
        (0, 2, 50, 20, 0, 1, 28, 5, 0)
    ])
    mean = a.mean()

    print(a[a < mean])


def instructional_code_19():
    a = np.array([
        (1, 2, 3, 4, 5),
        (10, 20, 30, 40, 50)
    ], dtype=np.int32)
    print("Original array:\n", a)

    # Multiply ndarray a by 2
    print("\nMultiply a by 2:\n", 2 * a)   # the same as matrix calculation

    # a / 2
    print(a / 2.0)


if __name__ == "__main__":
    instructional_code_19()
