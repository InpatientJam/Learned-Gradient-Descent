import torch
import numpy as np


def normalize_p2d(joint2d, confidence=None, threshold=0.1):
    if confidence is None:
        confidence = torch.ones_like(joint2d)[:, :, 0]

    joint2d_clone = joint2d.clone()

    # ignore low conf joint
    mask = confidence < threshold
    joint2d_clone[mask, :] = torch.finfo(joint2d_clone.dtype).max
    joint_min = joint2d_clone.min(dim=1).values.unsqueeze(1)
    joint2d_clone[mask, :] = torch.finfo(joint2d_clone.dtype).min
    joint_max = joint2d_clone.max(dim=1).values.unsqueeze(1)

    # scale the diffence to 1
    scale = (joint_max - joint_min).max(dim=2).values.unsqueeze(-1)
    offset = 0.5 * (joint_max + joint_min)

    # normalize joint to [0,1]
    joint2d = (joint2d - offset) / scale + 0.5

    # set low conf joint to 0.5
    joint2d[mask] = 0.5
    return joint2d


def compute_similarity_transform(X, Y, compute_optimal_scale=True):
    """
    A port of MATLAB's `procrustes` function to Numpy.
    Adapted from http://stackoverflow.com/a/18927641/1884420

    Args
      X: array NxM of targets, with N number of points and M point dimensionality
      Y: array NxM of inputs
      compute_optimal_scale: whether we compute optimal scale or force it to be 1

    Returns:
      d: squared error after transformation
      Z: transformed Y
      T: computed rotation
      b: scaling
      c: translation
    """
    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 = X0 / normX
    Y0 = Y0 / normY

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    # Make sure we have a rotation
    detT = np.linalg.det(T)
    V[:, -1] *= np.sign(detT)
    s[-1] *= np.sign(detT)
    T = np.dot(V, U.T)

    traceTA = s.sum()

    if compute_optimal_scale:  # Compute optimum scaling of Y.
        b = traceTA * normX / normY
        d = 1 - traceTA**2
        Z = normX * traceTA * np.dot(Y0, T) + muX
    else:  # If no scaling allowed
        b = 1
        d = 1 + ssY / ssX - 2 * traceTA * normY / normX
        Z = normY * np.dot(Y0, T) + muX

    c = muX - b * np.dot(muY, T)

    return d, Z, T, b, c


def compute_MPJPE(joint3d_pred, joint3d_real, PA=True):
    errors = []
    for j in range(joint3d_pred.shape[0]):
        joint3d_real_ = joint3d_real.data[j].cpu().numpy()
        joint3d_pred_ = joint3d_pred.data[j].cpu().numpy()
        if PA:
            d1, joint3d_pred_, T, b, c = compute_similarity_transform(
                joint3d_real_, joint3d_pred_)
        d = np.sqrt(((joint3d_pred_ - joint3d_real_)**2).sum(-1)).mean()
        errors.append(d)
    return errors
