from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import torch
import numpy as np
from skimage.restoration import denoise_tv_chambolle, denoise_tv_bregman
from skimage.util import random_noise
from scipy.optimize import minimize
from skimage.util import img_as_float
from skimage import color

try:
    from lib.transformations.tv_bregman import _denoise_tv_bregman
except ImportError:
    raise ImportError("tv_bregman not found. Check build script")


def tv(x, p):
    f = np.linalg.norm(x[1:, :] - x[:-1, :], p, axis=1).sum()
    f += np.linalg.norm(x[:, 1:] - x[:, :-1], p, axis=0).sum()
    return f


def tv_dx(x, p):
    if p == 1:
        x_diff0 = np.sign(x[1:, :] - x[:-1, :])
        x_diff1 = np.sign(x[:, 1:] - x[:, :-1])
    elif p > 1:
        x_diff0_norm = np.power(np.linalg.norm(x[1:, :] - x[:-1, :], p, axis=1), p - 1)
        x_diff1_norm = np.power(np.linalg.norm(x[:, 1:] - x[:, :-1], p, axis=0), p - 1)
        x_diff0_norm[x_diff0_norm < 1e-3] = 1e-3
        x_diff1_norm[x_diff1_norm < 1e-3] = 1e-3
        x_diff0_norm = np.repeat(x_diff0_norm[:, np.newaxis], x.shape[1], axis=1)
        x_diff1_norm = np.repeat(x_diff1_norm[np.newaxis, :], x.shape[0], axis=0)
        x_diff0 = p * np.power(x[1:, :] - x[:-1, :], p - 1) / x_diff0_norm
        x_diff1 = p * np.power(x[:, 1:] - x[:, :-1], p - 1) / x_diff1_norm
    df = np.zeros(x.shape)
    df[:-1, :] = -x_diff0
    df[1:, :] += x_diff0
    df[:, :-1] -= x_diff1
    df[:, 1:] += x_diff1
    return df


def tv_l2(x, y, w, lam, p):
    f = 0.5 * np.power(x - y.flatten(), 2).dot(w.flatten())
    x = np.reshape(x, y.shape)
    return f + lam * tv(x, p)


def tv_l2_dx(x, y, w, lam, p):
    x = np.reshape(x, y.shape)
    df = (x - y) * w
    return df.flatten() + lam * tv_dx(x, p).flatten()


def tv_inf(x, y, lam, p, tau):
    x = np.reshape(x, y.shape)
    return tau + lam * tv(x, p)


def tv_inf_dx(x, y, lam, p, tau):
    x = np.reshape(x, y.shape)
    return lam * tv_dx(x, p).flatten()


def minimize_tv(img, w, lam=0.01, p=2, solver='L-BFGS-B', maxiter=100, verbose=False):
    x_opt = np.copy(img)
    if solver == 'L-BFGS-B' or solver == 'CG' or solver == 'Newton-CG':
        for i in range(img.shape[2]):
            options = {'disp': verbose, 'maxiter': maxiter}
            res = minimize(
                tv_l2, x_opt[:, :, i], (img[:, :, i], w[:, :, i], lam, p),
                method=solver, jac=tv_l2_dx, options=options).x
            x_opt[:, :, i] = np.reshape(res, x_opt[:, :, i].shape)
    else:
        print('unsupported solver ' + solver)
        exit()
    return x_opt


def minimize_tv_inf(img, w, tau=0.1, lam=0.01, p=2, solver='L-BFGS-B', maxiter=100,
                    verbose=False):
    x_opt = np.copy(img)
    if solver == 'L-BFGS-B' or solver == 'CG' or solver == 'Newton-CG':
        for i in range(img.shape[2]):
            options = {'disp': verbose, 'maxiter': maxiter}
            lower = img[:, :, i] - tau
            upper = img[:, :, i] + tau
            lower[w[:, :, i] < 1e-6] = 0
            upper[w[:, :, i] < 1e-6] = 1
            bounds = np.array([lower.flatten(), upper.flatten()]).transpose()
            res = minimize(
                tv_inf, x_opt[:, :, i], (img[:, :, i], lam, p, tau),
                method=solver, bounds=bounds, jac=tv_inf_dx, options=options).x
            x_opt[:, :, i] = np.reshape(res, x_opt[:, :, i].shape)
    else:
        print('unsupported solver ' + solver)
        exit()
    return x_opt


def minimize_tv_bregman(img, mask, weight, maxiter=100, gsiter=10, eps=0.001,
                        isotropic=True):
    img = img_as_float(img)
    mask = mask.astype('uint8', order='C')
    return _denoise_tv_bregman(img, mask, weight, maxiter, gsiter, eps, isotropic)


# applies TV reconstruction
def reconstruct(img, drop_rate, recons, weight, drop_rate_post=0, lab=False,
                verbose=False, input_filepath=''):
    assert torch.is_tensor(img)
    temp = np.rollaxis(img.numpy(), 0, 3)
    w = np.ones_like(temp)
    if drop_rate > 0:
        # independent channel/pixel salt and pepper
        temp2 = random_noise(temp, 's&p', amount=drop_rate, salt_vs_pepper=0)
        # per-pixel all channel salt and pepper
        r = temp2 - temp
        w = (np.absolute(r) < 1e-6).astype('float')
        temp = temp + r
    if lab:
        temp = color.rgb2lab(temp)
    if recons == 'none':
        temp = temp
    elif recons == 'chambolle':
        temp = denoise_tv_chambolle(temp, weight=weight, multichannel=True)
    elif recons == 'bregman':
        if drop_rate == 0:
            temp = denoise_tv_bregman(temp, weight=1 / weight, isotropic=True)
        else:
            temp = minimize_tv_bregman(
                temp, w, weight=1 / weight, gsiter=10, eps=0.01, isotropic=True)
    elif recons == 'tvl2':
        temp = minimize_tv(temp, w, lam=weight, p=2, solver='L-BFGS-B', verbose=verbose)
    elif recons == 'tvinf':
        temp = minimize_tv_inf(
            temp, w, tau=weight, p=2, solver='L-BFGS-B', verbose=verbose)
    else:
        print('unsupported reconstruction method ' + recons)
        exit()
    if lab:
        temp = color.lab2rgb(temp)
    # temp = random_noise(temp, 's&p', amount=drop_rate_post, salt_vs_pepper=0)
    temp = torch.from_numpy(np.rollaxis(temp, 2, 0)).float()
    return temp
