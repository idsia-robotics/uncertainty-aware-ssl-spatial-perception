import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# 3d


class PosePlot():
    def __init__(self, ax, mat, name, scale=.1):
        self.ax = ax
        self.mat = mat
        self.name = name

        if isinstance(scale, (int, float)):
            scale = [scale] * 3

        self.scale = scale

        self.draw_pose()

    def compute(self):
        self.xhat = np.dot(self.mat, np.array(
            [[0, 0, 0, 1], [self.scale[0], 0, 0, 1]]).T)
        self.yhat = np.dot(self.mat, np.array(
            [[0, 0, 0, 1], [0, self.scale[1], 0, 1]]).T)
        self.zhat = np.dot(self.mat, np.array(
            [[0, 0, 0, 1], [0, 0, self.scale[2], 1]]).T)

    def update(self, mat=None, name=None):
        if mat is not None:
            self.mat = mat
            self.compute()

            self.dxhat.set_data(self.xhat[0, :], self.xhat[1, :])
            self.dxhat.set_3d_properties(self.xhat[2, :])
            self.dyhat.set_data(self.yhat[0, :], self.yhat[1, :])
            self.dyhat.set_3d_properties(self.yhat[2, :])
            self.dzhat.set_data(self.zhat[0, :], self.zhat[1, :])
            self.dzhat.set_3d_properties(self.zhat[2, :])
            self.text.set_position((self.xhat[0, 0], self.xhat[1, 0]))
            self.text.set_3d_properties(self.xhat[2, 0], zdir='x')

        if name is not None:
            self.name = name
            self.text.set_text(name)

    def draw_pose(self):
        self.compute()

        self.text = self.ax.text(self.xhat[0, 0], self.xhat[1, 0], self.xhat[2, 0],
                                 self.name, va='top', ha='center')
        self.dxhat = self.ax.plot(
            self.xhat[0, :], self.xhat[1, :], self.xhat[2, :], 'r-')[0]
        self.dyhat = self.ax.plot(
            self.yhat[0, :], self.yhat[1, :], self.yhat[2, :], 'g-')[0]
        self.dzhat = self.ax.plot(
            self.zhat[0, :], self.zhat[1, :], self.zhat[2, :], 'b-')[0]

        self.drawables = [self.dxhat, self.dyhat, self.dzhat, self.text]


def project(transf_matrix, point=[0, 0, 0, 1]):
    '''Projects a point using a transformation matrix into a 2D plane.

    Args:
            transf_matrix: a 4x4 homogeneous transformation matrix.
            point: a XYZ poin having 4x1 shape and whose last value is 1.

    Returns:
            the XY coordinates of the point projected onto the 2D plane.
    '''
    P = np.array([[462.1379497504639, 0.0, 320.5, -0.0],
                  [0.0, 462.1379497504639, 240.5, 0.0],
                  [0.0, 0.0, 1.0, 0.0]])
    res = np.dot(P, np.dot(transf_matrix, point))
    res = res[:-1:] / res[-1]
    return res


# 2d


class FramePlot():
    def __init__(self, ax, mat, colors=['r', 'g', 'b']):
        self.ax = ax
        self.mat = mat
        self.colors = colors

        self.project_frame()

    def compute(self):
        pt = np.array([0, 0, 0, 1])
        self.o2d = project(self.mat, pt)
        self.i2d = project(self.mat, pt + [.05, 0, 0, 0])
        self.j2d = project(self.mat, pt + [0, .05, 0, 0])
        self.k2d = project(self.mat, pt + [0, 0, .05, 0])

    def update(self, mat):
        self.mat = mat
        self.compute()

        self.di.set_data(np.array([self.o2d[0], self.i2d[0]]),
                         np.array([self.o2d[1], self.i2d[1]]))
        self.dj.set_data(np.array([self.o2d[0], self.j2d[0]]),
                         np.array([self.o2d[1], self.j2d[1]]))
        self.dk.set_data(np.array([self.o2d[0], self.k2d[0]]),
                         np.array([self.o2d[1], self.k2d[1]]))

    def project_frame(self):
        self.compute()

        self.di = self.ax.plot([self.o2d[0], self.i2d[0]], [self.o2d[1], self.i2d[1]],
                               linewidth=1, c=self.colors[0])[0]
        self.dj = self.ax.plot([self.o2d[0], self.j2d[0]], [self.o2d[1], self.j2d[1]],
                               linewidth=1, c=self.colors[1])[0]
        self.dk = self.ax.plot([self.o2d[0], self.k2d[0]], [self.o2d[1], self.k2d[1]],
                               linewidth=1, c=self.colors[2])[0]

        self.drawables = [self.di, self.dj, self.dk]


def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
