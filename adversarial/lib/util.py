# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import tempfile

import torch

# constants:
CHECKPOINT_FILE = 'checkpoint.torch'


# function that measures top-k accuracy:
def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100. / batch_size))
    return res


# function that tries to load a checkpoint:
def load_checkpoint(checkpoint_folder):

    # read what the latest model file is:
    filename = os.path.join(checkpoint_folder, CHECKPOINT_FILE)
    if not os.path.exists(filename):
        return None

    # load and return the checkpoint:
    return torch.load(filename)


# function that saves checkpoint:
def save_checkpoint(checkpoint_folder, state):

    # make sure that we have a checkpoint folder:
    if not os.path.isdir(checkpoint_folder):
        try:
            os.makedirs(checkpoint_folder)
        except BaseException:
            print('| WARNING: could not create directory %s' % checkpoint_folder)
    if not os.path.isdir(checkpoint_folder):
        return False

    # write checkpoint atomically:
    try:
        with tempfile.NamedTemporaryFile(
                'w', dir=checkpoint_folder, delete=False) as fwrite:
            tmp_filename = fwrite.name
            torch.save(state, fwrite.name)
        os.rename(tmp_filename, os.path.join(checkpoint_folder, CHECKPOINT_FILE))
        return True
    except BaseException:
        print('| WARNING: could not write checkpoint to %s.' % checkpoint_folder)
        return False


# function that adjusts the learning rate:
def adjust_learning_rate(base_lr, epoch, optimizer, lr_decay, lr_decay_stepsize):
    lr = base_lr * (lr_decay ** (epoch // lr_decay_stepsize))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# adversary functions
# computes SSIM for a single block
def SSIM(x, y):
    x = x.resize_(x.size(0), x.size(1) * x.size(2) * x.size(3))
    y = y.resize_(y.size(0), y.size(1) * y.size(2) * y.size(3))
    N = x.size(1)
    mu_x = x.mean(1)
    mu_y = y.mean(1)
    sigma_x = x.std(1)
    sigma_y = y.std(1)
    sigma_xy = ((x - mu_x.expand_as(x)) * (y - mu_y.expand_as(y))).sum(1) / (N - 1)
    ssim = (2 * mu_x * mu_y) * (2 * sigma_xy)
    ssim = ssim / (mu_x.pow(2) + mu_y.pow(2))
    ssim = ssim / (sigma_x.pow(2) + sigma_y.pow(2))
    return ssim


# mean SSIM using local block averaging
def MSSIM(x, y, window_size=16, stride=4):
    ssim = torch.zeros(x.size(0))
    L = x.size(2)
    W = x.size(3)
    x_inds = torch.arange(0, L - window_size + 1, stride).long()
    y_inds = torch.arange(0, W - window_size + 1, stride).long()
    for i in x_inds:
        for j in y_inds:
            x_sub = x[:, :, i:(i + window_size), j:(j + window_size)]
            y_sub = y[:, :, i:(i + window_size), j:(j + window_size)]
            ssim = ssim + SSIM(x_sub, y_sub)
    return ssim / x_inds.size(0) / y_inds.size(0)


# forwards input through model to get probabilities
def get_probs(model, imgs, output_prob=False):
    softmax = torch.nn.Softmax(1)
    # probs = torch.zeros(imgs.size(0), n_classes)
    imgsvar = torch.autograd.Variable(imgs.squeeze(), volatile=True)
    output = model(imgsvar)
    if output_prob:
        probs = output.data.cpu()
    else:
        probs = softmax.forward(output).data.cpu()

    return probs


# calls get_probs to get predictions
def get_labels(model, input, output_prob=False):
    probs = get_probs(model, input, output_prob)
    _, label = probs.max(1)
    return label.squeeze()
