import numpy as np


def filter_to_2d(filters):
    """Reshape (K,C,R,S) filter into (K, C*R*S).

    Args:
        filters (numpy.ndarray): (K,C,R,S) filter.

    Returns:
        numpy.ndarray: Reshaped (K, C*R*S) filter.

    Example:
    >>> import numpy as np
    >>> f = np.arange(24).reshape([2, 3, 2, 2])
    >>> filter_to_2d(f)
    array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11],
           [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]])
    """

    return filters.reshape(filters.shape[0], -1)


def data_to_2d(data, filters, stride_h, stride_w, padding_h, padding_w):
    """Reshape (N,C,H,W) data into (C*R*S, N*P*Q).

    Args:
        data       (numpy.ndarray): (N,C,H,W) data.
        filters    (numpy.ndarray): (K,C,R,S) filter.
        stride_h, stride_w   (int): vertical / horizontal strides.
        padidng_h, padding_w (int): vertical / horizontal strides.

    Returns:
        numpy.ndarray: Reshaped (C*R*S, N*P*Q) data.

    Example:
    >>> import numpy as np
    >>> data = np.arange(54).reshape([2, 3, 3, 3])
    >>> filters = np.arange(24).reshape([2, 3, 2, 2])
    >>> data_to_2d(data, filters, 1, 1, 0, 0)
    array([[ 0,  1,  3,  4, 27, 28, 30, 31],
           [ 1,  2,  4,  5, 28, 29, 31, 32],
           [ 3,  4,  6,  7, 30, 31, 33, 34],
           [ 4,  5,  7,  8, 31, 32, 34, 35],
           [ 9, 10, 12, 13, 36, 37, 39, 40],
           [10, 11, 13, 14, 37, 38, 40, 41],
           [12, 13, 15, 16, 39, 40, 42, 43],
           [13, 14, 16, 17, 40, 41, 43, 44],
           [18, 19, 21, 22, 45, 46, 48, 49],
           [19, 20, 22, 23, 46, 47, 49, 50],
           [21, 22, 24, 25, 48, 49, 51, 52],
           [22, 23, 25, 26, 49, 50, 52, 53]])
    """

    _, C, R, S = filters.shape

    data_padded = np.pad(data, ((0, 0), (0, 0), (padding_h, padding_h), (padding_w, padding_w)), mode='constant')
    k, i, j = get_im2col_indices(data.shape, filters.shape, stride_h, stride_w, padding_h, padding_w)

    cols = np.concatenate(data_padded[:, k, i, j], axis=1)
    return cols


def filter2d_to_orig(filter_2d, shape):
    """Reshape (K, C*R*S) filter into (K, C, R, S).

    Args:
        filter_2d (numpy.ndarray): (K, C*R*S) filter.
        shape        ((int, int)): (R, S)

    Returns:
        numpy.ndarray: Reshaped (K, C, R, S) filter.

    Example:
    >>> import numpy as np
    >>> f = np.arange(24).reshape([2, 3, 2, 2])
    >>> filter_2d = filter_to_2d(f)
    >>> np.array_equal(f, filter2d_to_orig(filter_2d, (2, 2)))
    True
    """

    R, S = shape
    return filter_2d.reshape(filter_2d.shape[0], -1, R, S)


def data2d_to_orig(data_2d, data_shape, filter_shape, stride_h, stride_w, padding_h, padding_w):
    """Reshape (C*R*S, N*P*Q) data into (N,C,H,W).

    Args:
        data_2d           (numpy.ndarray): (C*R*S, N*P*Q) data.
        data_shape ((int, int, int, int)): (N,C,H,W)
        filter_shape         ((int, int)): (R,S)
        stride_h, strite_w          (int): v / h strides.
        padding_h, padding_w        (int): v / h paddings.

    Returns:
        numppy.ndarray: Reshaped (N,C,H,W) data.

    Example:
    >>> import numpy as np
    >>> data = np.arange(54).reshape([2,3,3,3])
    >>> filters = np.arange(24).reshape([2,3,2,2])
    >>> data_2d = data_to_2d(data, filters, 1, 1, 0, 0)
    >>> np.array_equal(data, data2d_to_orig(data_2d, data.shape, (2, 2), 1, 1, 0, 0))
    True
    """

    N, C, H, W = data_shape
    R, S = filter_shape
    H_padded, W_padded = H + 2 * padding_h, W + 2 * padding_w
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=data_2d.dtype)
    k, i, j = get_im2col_indices(data_shape, (0, 0, R, S), stride_h, stride_w, padding_h, padding_w)

    cols = np.hsplit(data_2d, N)
    x_padded[:, k, i, j] += cols

    if padding_h == 0 and padding_w == 0:
        return x_padded
    return x_padded[:, :, padding_w:-padding_w, padding_h:-padding_h]


# Taken from https://github.com/huyouare/CS231n/blob/master/assignment2/cs231n/im2col.py
def get_im2col_indices(x_shape, filters_shape, stride_h, stride_w, padding_h, padding_w):
    N, C, H, W = x_shape
    _, _, R, S = filters_shape

    # calculate output shape P, Q
    assert (H - R + 2 * padding_h) % stride_h == 0
    assert (H - S + 2 * padding_w) % stride_w == 0
    P = (H - R + 2 * padding_h) / stride_h + 1
    Q = (W - S + 2 * padding_w) / stride_w + 1

    i0 = np.repeat(np.arange(R), S)
    i0 = np.tile(i0, C)
    i1 = stride_h * np.repeat(np.arange(P), Q)
    j0 = np.tile(np.arange(S), R * C)
    j1 = stride_w * np.tile(np.arange(Q), P)

    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), R * S).reshape(-1, 1)

    return (k, i, j)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
