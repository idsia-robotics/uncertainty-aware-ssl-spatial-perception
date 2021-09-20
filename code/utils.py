import cv2
import math
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
from tf.transformations import quaternion_from_euler, translation_matrix, quaternion_matrix, decompose_matrix


def jpeg_to_np(image, size=None, normalize=False, eps=1e-7):
    '''Converts a jpeg image in a 2d numpy array of BGR pixels and resizes it to the given size (if provided).

    Args:
            image: a compressed BGR jpeg image.
            size: a tuple containing width and height, or None for no resizing.
            normalize: a boolean flag representing wether or not to normalize the image.

    Returns:
            the raw, resized image as a 2d numpy array of RGB pixels.
    '''
    compressed = np.fromstring(image, np.uint8)
    raw = cv2.imdecode(compressed, cv2.IMREAD_COLOR)
    raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB).astype(np.float32)

    if size:
        raw = cv2.resize(raw, size)

    if normalize:
        raw = (raw - raw.min()) / (raw.max() - raw.min() + eps)

    return raw


def transf_matrix(pose):
    '''Converts a SE(3) pose stored in XYZQuat format to the respective homogeneous transformation matrix.

    Args:
            pose: a XYZQuat array-like pose.

    Returns:
            the 4x4 homogeneous transformation matrix.
    '''
    t = translation_matrix(pose[:3])
    r = quaternion_matrix(pose[3:])
    return np.dot(t, r)


def se3_pose(transf_matrix):
    '''Converts a homogeneous transformation matrix to the respective SE(3) pose stored in XYZQuat format.

    Args:
            transf_matrix: a 4x4 homogeneous transformation matrix.

    Returns:
            the XYZQuat ndarray pose.
    '''
    _, _, rpy, xyz, _ = decompose_matrix(transf_matrix)
    rpy = quaternion_from_euler(*rpy, axes='sxyz')
    rpy /= np.linalg.norm(rpy)
    return np.array(xyz.tolist() + rpy.tolist())


def to_tensor(xy, device='cpu', dtype=torch.float):
    '''Converts a XY tuple of numpy arrays to a tuple of torch tensors.

    Args:
            xy: a tuple of numpy arrays.
            device: device on which to perform computations (usually "cuda" or "cpu").
            dtype: a torch numeric type.

    Returns:
            the XY tuple of torch tensors.
    '''
    def internal(x):
        x = np.ascontiguousarray(x)
        x = torch.tensor(x, device=device, dtype=dtype)
        return x

    return tuple(map(internal, xy))


def apply_albumentations(xy, transform):
    '''Applies albumentations on the x of the XY tuple.

    Args:
            xy: a tuple of numpy arrays.
            transform: a function composition of albumnetations to be applied on x.

    Returns:
            the XY tuple with x augmented.
    '''
    x, y = xy
    is_batched = x.ndim == 4

    if is_batched:
        for i in range(x.shape[0]):
            x[i] = transform(image=x[i])['image']
    else:
        x = transform(image=x)['image']

    return (x, y)


def reorder_dims(xy):
    '''Changes the order of the dimensions of x from HWC to CHW.

    Args:
            xy: a tuple of numpy arrays.

    Returns:
            the XY tuple with x dimensions ordered as CHW.
    '''
    x, y = xy
    is_batched = x.ndim == 4
    x = np.moveaxis(x, -1, 1 * is_batched)
    return (x, y)


def d_quat(p, q, eps=1e-7):
    '''Computes the quaternionic distance of two quaternions.

    Args:
            p: a quaternion.
            q: a quaternion.

    Returns:
            the quaternionic distance of two quaternions.
    '''
    abs_dot = torch.sum(p * q, dim=1, keepdim=True).abs()
    abs_dot = torch.clamp_max(abs_dot, 1 - eps)
    return 2 / np.pi * torch.acos(abs_dot)


class SE3Loss(torch.nn.Module):
    def __init__(self, lmbd=10, eps=1e-7,
                 pos_loss=torch.nn.functional.mse_loss):
        super(SE3Loss, self).__init__()
        self.eps = eps
        self.lmbd = lmbd
        self.position_loss = pos_loss

    def forward(self, y_pred, y_true, normalize=True):
        pos_pred = y_pred[..., :3]
        pos_true = y_true[..., :3]
        rot_pred = y_pred[..., 3:] + self.eps
        rot_true = y_true[..., 3:] + self.eps

        if normalize:
            rot_pred = rot_pred / torch.norm(rot_pred, dim=1, keepdim=True)
            rot_true = rot_true / torch.norm(rot_true, dim=1, keepdim=True)

        pos_loss = self.position_loss(pos_pred, pos_true)
        rot_loss = d_quat(rot_pred, rot_true).mean()

        return torch.stack([self.lmbd * pos_loss, rot_loss])
