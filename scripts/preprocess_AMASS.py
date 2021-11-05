"""
Code to extract SMPL parameters from SMPLH
"""
import numpy as np
import glob


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

def swap_axes(thetas):
    mat = euler_to_rot([-1.57], [-1.57], [0])
    global_rotation = batch_rodrigues(thetas[:, :3])
    thetas[:, :3] = batch_rot_to_angle_axis(mat @ global_rotation) 
    return thetas


def preprocess(dataset_path, out_file):
    # data we'll save
    output = {
        "smpl_pose": [],
        "smpl_shape": [],
    }

    print("Start processing")
    for filename in sorted(glob.glob(f"{dataset_path}/*/*/*.npz")):
        data = np.load(filename)
        if "poses" not in data.keys() or "betas" not in data.keys():
            print("-- skipping", filename)
            continue
        else:
            print("-- processing", filename)

        # to convert SMPL+H => SMPL
        # 1. use first 10 shape params
        # 2. use first 66 pose params
        smpl_pose = data["poses"][:, :72]
        smpl_pose[:, 66:] = 0
        smpl_shape = np.tile(data["betas"][:10], (len(smpl_pose), 1))

        # swap axes
        smpl_pose = swap_axes(smpl_pose)

        output["smpl_shape"].append(smpl_shape.astype(np.float32))
        output["smpl_pose"].append(smpl_pose.astype(np.float32))

    for k, v in output.items():
        output[k] = np.concatenate(v, axis=0)
        print(k, output[k].shape)
    np.savez(out_file, **output)


if __name__ == "__main__":
    preprocess("./data/AMASS/", "./data/AMASS.npz")
