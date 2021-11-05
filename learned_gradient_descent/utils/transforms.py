import numpy as np


def rotate_axis(angles, perturbation):
    rot_mat = batch_rodrigues(angles)
    rot_mat = euler_to_rot(perturbation[:, 0],
                           perturbation[:, 1],
                           perturbation[:, 2]) @ rot_mat
    angles = batch_rot_to_angle_axis(rot_mat)
    return angles


def euler_to_rot(x, y, z):
    """convert (row, pitch, yaw) to rotation matrix

    Args:
        x (B, ): row
        y (B, ): pitch
        z (B, ): yaw

    Return:
        (B, 3, 3)
    """
    zeros = np.zeros_like(z)
    ones = np.ones_like(z)
    R = np.eye(3).reshape(1, 3, 3)
    for i, angle in enumerate([z, y, x]):
        cos, sin = np.cos(angle), np.sin(angle)
        if i == 0:
            Ri = np.stack([cos, -sin, zeros, sin, cos,
                          zeros, zeros, zeros, ones], axis=1)
        elif i == 1:
            Ri = np.stack([cos, zeros, sin, zeros, ones,
                          zeros, -sin, zeros, cos], axis=1)
        else:
            Ri = np.stack([ones, zeros, zeros, zeros,
                          cos, -sin, zeros, sin, cos], axis=1)
        R = R @ Ri.reshape(-1, 3, 3)
    return R


def batch_rodrigues(rvecs):
    """convert angle axis repr to rotation matrices

    Args:
        rvecs (B, 3): rotation vectors

    Returns:
        (B, 3, 3)
    """
    norm = np.sqrt(np.square(rvecs + 1e-8).sum(axis=1)).reshape(-1, 1)
    rvecs = rvecs / norm

    z = np.zeros_like(norm)
    rx, ry, rz = np.split(rvecs, 3, axis=1)
    K = np.stack((z, -rz, ry, rz, z, -rx, -ry, rx, z),
                 axis=1).reshape(-1, 3, 3)

    sin = np.sin(norm).reshape(-1, 1, 1)
    cos = np.cos(norm).reshape(-1, 1, 1)
    return np.eye(3).reshape(1, 3, 3) + sin * K + (1 - cos) * np.matmul(K, K)


def batch_skew_matrix(k):
    zeros = np.zeros(len(k))
    return np.stack([zeros, -k[:, 2], k[:, 1],
                     k[:, 2], zeros, -k[:, 0],
                     -k[:, 1], k[:, 0], zeros], axis=1).reshape(-1, 3, 3)


def batch_rot_to_angle_axis(R):
    """convert rotation matrices to angle axis repr"""
    # R @ u = 0 => u is the axis
    u = np.stack([R[:, 2, 1] - R[:, 1, 2],
                  R[:, 0, 2] - R[:, 2, 0],
                  R[:, 1, 0] - R[:, 0, 1]], axis=1)
    u /= np.linalg.norm(u, axis=1)[:, None]

    # trace(R) = 2cos + 1 => |theta| = arccos((tr(R) - 1) / 2)
    cos = (np.trace(R, axis1=1, axis2=2) - 1) / 2

    # R - R.T = 2 * skew(u) * sin(theta)
    S = (R - R.transpose(0, 2, 1)) / 2
    sin = (np.trace(S, offset=2, axis1=1, axis2=2) +
           np.trace(S, offset=-1, axis1=1, axis2=2)) / u.sum(axis=1)
    theta = np.arctan2(sin, cos)
    return u * theta[:, None]


class RandomRotateSMPL(object):
    def __init__(self):
        self.rot_lb = np.array([-20, 0, -20]) / 180. * np.pi
        self.rot_ub = np.array([20, 360, 20]) / 180. * np.pi

    def __call__(self, batch):
        """
        Args:
            batch = {
                "betas":  (10),
                "thetas": (72),
            }
        """
        # convert to batch
        betas = batch["betas"][None, :]
        thetas = batch["thetas"][None, :]
        angle = np.random.uniform(self.rot_lb, self.rot_ub, size=(1, 3))
        thetas[:, :3] = rotate_axis(thetas[:, :3], angle)

        # convert from batch
        return {
            "betas": betas[0],
            "thetas": thetas[0]
        }
