import numpy as np


def quaternion_transform(r):
    """
    Get optimal rotation
    note: translation will be zero when the centroids of each molecule are the
    same
    """
    Wt_r = makeW(*r).T
    Q_r = makeQ(*r)
    rot = Wt_r.dot(Q_r)[:3, :3]
    return rot


def makeW(r1, r2, r3, r4=0):
    """
    matrix involved in quaternion rotation
    """
    W = np.asarray(
        [
            [r4, r3, -r2, r1],
            [-r3, r4, r1, r2],
            [r2, -r1, r4, r3],
            [-r1, -r2, -r3, r4],
        ]
    )
    return W


def makeQ(r1, r2, r3, r4=0):
    """
    matrix involved in quaternion rotation
    """
    Q = np.asarray(
        [
            [r4, -r3, r2, r1],
            [r3, r4, -r1, r2],
            [-r2, r1, r4, r3],
            [-r1, -r2, -r3, r4],
        ]
    )
    return Q


def quaternion_rotate(X, Y):
    """
    Calculate the rotation
    Parameters
    ----------
    X: np.ndarray
        (N,D) matrix, where N is points and D is dimension.
    Y: np.ndarray
        (N,D) matrix, where N is points and D is dimension.
    Returns
    -------
    rot : matrix
        Rotation matrix (D,D)
    """
    N = X.shape[0]
    W = np.asarray([makeW(*Y[k]) for k in range(N)])
    Q = np.asarray([makeQ(*X[k]) for k in range(N)])
    Qt_dot_W = np.asarray([np.dot(Q[k].T, W[k]) for k in range(N)])
    # NOTE UNUSED W_minus_Q = np.asarray([W[k] - Q[k] for k in range(N)])
    A = np.sum(Qt_dot_W, axis=0)
    eigen = np.linalg.eigh(A)
    r = eigen[1][:, eigen[0].argmax()]
    rot = quaternion_transform(r)
    return rot
