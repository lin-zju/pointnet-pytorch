import numpy as np
import matplotlib.pyplot as plt
import math
from lib.datasets.modelnet40 import ModelNet40
from lib.utils.config import cfg


def get2Dpoints(point_cloud, K, R, T):
    """
    :param point_cloud: (3, N)
    """
    E = np.hstack([R, T])
    M = K.dot(E)
    
    N = point_cloud.shape[1]
    cloud_homo = np.vstack([point_cloud, np.ones(N)])
    pts2D_homo = M.dot(cloud_homo)
    pts2D = pts2D_homo[:2] / pts2D_homo[2]
    return pts2D


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    theta = theta / 180.0 * math.pi
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def visualize(cloud, K, R, T):
    pts = get2Dpoints(cloud, K, R, T)
    plt.scatter(pts[0], pts[1], s=1)


def visualize_id(dataset, id, K, R, T):
    dataset = ModelNet40(cfg.MODELNET)
    rows = math.ceil(math.sqrt(4))
    count = 0
    for data in dataset:
        cloud, label = data[0].numpy(), data[1].numpy()
        if label == id:
            count += 1
            plt.subplot(2, 2, count)
            visualize(cloud, K, R, T)
            if count == 4:
                break
    plt.show()


def visualize_categories(dataset, K, R, T):
    dataset = ModelNet40(cfg.MODELNET)
    for i in range(40):
        for data in dataset:
            cloud, label = data[0].numpy(), data[1].numpy()
            if label == i:
                plt.subplot(4, 4, i + 1)
                visualize(cloud, K, R, T)
    
    plt.show()


if __name__ == '__main__':
    K = np.array(
        [[128, 0, 128],
         [0, 128, 128],
         [0, 0, 1]])
    
    rm = rotation_matrix
    # R = np.eye(3)
    R = rm([0, 0, 1], 0).dot(rm([0, 1, 0], 45)).dot(rm([1, 0, 0], -45))
    T = np.array([0, 0, 2]).reshape(3, 1)
    dataset = ModelNet40(cfg.MODELNET)
    
    for i in range(40):
        plt.figure(i + 1)
        plt.suptitle('category {}'.format(i))
        visualize_id(dataset, i, K, R, T)
