import math
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R


def v2m(v):
    """ 7-element pose vector to 3d homogeneous pose matrix """
    return np.vstack((
        np.hstack(
            (R.from_quat(v[3:]).as_matrix(), v[:3, np.newaxis])),
        np.array([[0, 0, 0, 1]]),
    ))


def rotnoise(sigma):
    """ Returns random rotations around z axis, gaussian distributed with given sigma [radians] """
    ret = np.eye(4)
    ret[:3, :3] = R.from_rotvec(
        np.array([0, 0, 1]) * np.random.normal(scale=sigma)
    ).as_matrix()
    return ret


def relrotnoise(dp, fraction):
    """Generates a rotation noise with sigma equal to a given fraction of the true rotation"""
    dtheta = R.from_matrix(dp[:3, :3]).as_euler("zyx")[0]
    return rotnoise(fraction * np.abs(dtheta))


class Odometry:
    def __init__(self, poses, noisemodel, Nnoises=5):
        self.Nnoises = Nnoises
        poses = list(poses)
        self.dposes = []
        self.ndposes = []
        for p0, p1 in zip(poses, poses[1:]):
            dpose = np.matmul(np.linalg.inv(p0), p1)
            self.dposes.append(dpose)
            ndpose = []
            self.ndposes.append(ndpose)
            for _ in range(self.Nnoises):
                ndpose.append(np.matmul(dpose, noisemodel(dpose)))

    def sample(self, from_ix, to_ix):
        assert(from_ix <= to_ix)
        assert(to_ix <= len(self.dposes))
        poses = [np.eye(4)]
        rand_ixs = np.random.randint(self.Nnoises, size=to_ix - from_ix)
        for rand_ix, ndps in zip(rand_ixs, self.ndposes[from_ix:to_ix]):
            poses.append(np.matmul(poses[-1], ndps[rand_ix]))
        return poses

    def sample_all(self):
        return self.sample(0, len(self.dposes))

    def __len__(self):
        return len(self.dposes)


def decompose_matrix(matrix, _EPS=1e-10):
    """Return sequence of transformations from transformation matrix.

    matrix : array_like
        Non-degenerative homogeneous transformation matrix

    Return tuple of:
        scale : vector of 3 scaling factors
        shear : list of shear factors for x-y, x-z, y-z axes
        angles : list of Euler angles about static x, y, z axes
        translate : translation vector along x, y, z axes
        perspective : perspective partition of matrix

    Raise ValueError if matrix is of wrong type or degenerative.

    >>> T0 = translation_matrix((1, 2, 3))
    >>> scale, shear, angles, trans, persp = decompose_matrix(T0)
    >>> T1 = translation_matrix(trans)
    >>> numpy.allclose(T0, T1)
    True
    >>> S = scale_matrix(0.123)
    >>> scale, shear, angles, trans, persp = decompose_matrix(S)
    >>> scale[0]
    0.123
    >>> R0 = euler_matrix(1, 2, 3)
    >>> scale, shear, angles, trans, persp = decompose_matrix(R0)
    >>> R1 = euler_matrix(*angles)
    >>> numpy.allclose(R0, R1)
    True

    """
    M = np.array(matrix, dtype=np.float64, copy=True).T
    if abs(M[3, 3]) < _EPS:
        raise ValueError("M[3, 3] is zero")
    M /= M[3, 3]
    P = M.copy()
    P[:, 3] = 0, 0, 0, 1
    if not np.linalg.det(P):
        raise ValueError("Matrix is singular")

    scale = np.zeros((3, ), dtype=np.float64)
    shear = [0, 0, 0]
    angles = [0, 0, 0]

    if any(abs(M[:3, 3]) > _EPS):
        perspective = np.dot(M[:, 3], np.linalg.inv(P.T))
        M[:, 3] = 0, 0, 0, 1
    else:
        perspective = np.array((0, 0, 0, 1), dtype=np.float64)

    translate = M[3, :3].copy()
    M[3, :3] = 0

    row = M[:3, :3].copy()
    scale[0] = np.linalg.norm(row[0])
    row[0] /= scale[0]
    shear[0] = np.dot(row[0], row[1])
    row[1] -= row[0] * shear[0]
    scale[1] = np.linalg.norm(row[1])
    row[1] /= scale[1]
    shear[0] /= scale[1]
    shear[1] = np.dot(row[0], row[2])
    row[2] -= row[0] * shear[1]
    shear[2] = np.dot(row[1], row[2])
    row[2] -= row[1] * shear[2]
    scale[2] = np.linalg.norm(row[2])
    row[2] /= scale[2]
    shear[1:] /= scale[2]

    if np.dot(row[0], np.cross(row[1], row[2])) < 0:
        scale *= -1
        row *= -1

    angles[1] = math.asin(-row[0, 2])
    if math.cos(angles[1]):
        angles[0] = math.atan2(row[1, 2], row[2, 2])
        angles[2] = math.atan2(row[0, 1], row[0, 0])
    else:
        #angles[0] = math.atan2(row[1, 0], row[1, 1])
        angles[0] = math.atan2(-row[2, 1], row[1, 1])
        angles[2] = 0.0

    return scale, shear, angles, translate, perspective


def m2v(m):
    _, _, rpy, xyz, _ = decompose_matrix(m)
    quat = R.from_euler('xyz', rpy).as_quat()
    quat /= np.linalg.norm(quat)
    return np.array(xyz.tolist() + quat.tolist())
