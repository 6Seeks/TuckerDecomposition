import numpy as np


def unfold(X, n):
    return np.reshape(np.moveaxis(X, n, 0), (X.shape[n], -1))


def refold(X, n, shape):
    shape = list(shape)
    mode_dim = shape.pop(n)
    shape.insert(0, mode_dim)
    return np.moveaxis(np.reshape(X, shape), 0, n)


def valid(G, As, X):

    _X = As[1]
    for i in np.arange(2, len(As)):
        _X = np.kron(_X, As[i])

    print("xx", np.allclose(refold(As[0]@unfold(G, 0)@_X.T, 0, X.shape), X))


def moden_product(X, Y, n):
    shape = list(X.shape)
    shape[n] = Y.shape[0]

    res = np.dot(Y, unfold(X, n))
    return refold(res, n, shape)


def HOSVD(X):
    # T tensor for tucker decompsition
    As = []
    for n in range(X.ndim):
        A, _v, _M = np.linalg.svd(unfold(X, n))
        As.append(A)
    # 计算核张量G
    G = X
    for i, A in enumerate(As):
        G = moden_product(G, A.T, i)
    valid(G, As, X)
    return G, As


def HOOI(X):
    _, As = HOSVD(X)
    for _ in range(100):
        for n in range(len(As)):
            Y = X
            for i, A in enumerate(As):
                if n == i:
                    continue
                Y = moden_product(Y, A.T, i)
            A, _v, _m = np.linalg.svd(unfold(Y, n))
            As[n] = A

    G = X
    for i, A in enumerate(As):
        G = moden_product(G, A.T, i)

    valid(G, As, X)
    return G, As


X = np.arange(2*3*4).reshape(2, 3, 4)
HOOI(X)
