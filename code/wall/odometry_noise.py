import math
import numpy as np


# Odometry functions


def m2v(m):
    xy = m[:-1, -1]
    angle = math.atan2(m[1, 0], m[0, 0])
    return np.array(xy.tolist() + [angle])


def mktr(x, y):
    return np.array([[1, 0, x],
                     [0, 1, y],
                     [0, 0, 1]])


def mkrot(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta), np.cos(theta), 0],
                     [0, 0, 1]])


def ddtr(vl, vr, l, dt):
    ''' returns the pose transform for a motion with duration dt of a differential
    drive robot with wheel speeds vl and vr and wheelbase l '''
    if(np.isclose(vl, vr)):  # we are moving straight, R is at the infinity and we handle this case separately
        return mktr((vr + vl) / 2 * dt, 0)  # note we translate along x ()
    omega = (vr - vl) / (2 * l)  # angular speed of the robot frame
    R = l * (vr + vl) / (vr - vl)
    # Make sure you understand this!
    return np.matmul(np.matmul(mktr(0, R), mkrot(omega * dt)), mktr(0, -R))


def parse_thymio_odometry(odom_aseba, noise=0.0):
    # Translates aseba units for each wheel's movement to meters.
    # encoder_to_displacement = 0.00031
    encoder_to_displacement = 0.00031 * np.array([[0.965, 0.948]])
    wheelbase = 0.0935 / 2  # half distance between wheels [m]
    dodom_aseba = np.diff(odom_aseba, axis=0)
    dodom_m = dodom_aseba * encoder_to_displacement
    dodom_m_noisy = dodom_m + dodom_m * np.random.randn(*dodom_m.shape) * noise
    poses = [np.eye(3)]
    for o in dodom_m_noisy:
        poses.append(np.matmul(poses[-1], ddtr(o[0], o[1], l=wheelbase, dt=1)))
    return poses
