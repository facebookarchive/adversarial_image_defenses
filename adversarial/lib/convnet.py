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

import progressbar

import torch
import torch.nn as nn
from lib.util import accuracy


# function that trains a model:
def train(model, criterion, optimizer, data_loader_hook=None,
                 start_epoch_hook=None, end_epoch_hook=None,
                 start_epoch=0, end_epoch=90, learning_rate=0.1):

    # assertions:
    assert isinstance(model, nn.Module)
    assert isinstance(criterion, nn.modules.loss._Loss)
    assert isinstance(optimizer, torch.optim.Optimizer)
    assert type(start_epoch) == int and start_epoch >= 0
    assert type(end_epoch) == int and end_epoch >= start_epoch
    assert type(learning_rate) == float and learning_rate > .0
    if start_epoch_hook is not None:
        assert callable(start_epoch_hook)
    if end_epoch_hook is not None:
        assert callable(end_epoch_hook)
    assert data_loader_hook is not None
    assert callable(data_loader_hook)

    # are we on CPU or GPU?
    is_gpu = not isinstance(model, torch.nn.backends.thnn.THNNFunctionBackend)

    # train the model:
    model.train()
    for epoch in range(start_epoch, end_epoch):

        data_loader = data_loader_hook(epoch)
        assert isinstance(data_loader, torch.utils.data.dataloader.DataLoader)

        # start-of-epoch hook:
        if start_epoch_hook is not None:
            start_epoch_hook(epoch, model, optimizer)

        # loop over training data:
        model.train()
        precs1, precs5, num_batches, num_total = [], [], 0, 0
        bar = progressbar.ProgressBar(len(data_loader))
        bar.start()
        for num_batches, (imgs, targets) in enumerate(data_loader):

            # copy data to GPU:
            if is_gpu:
                cpu_targets = targets.clone()
                targets = targets.cuda(async=True)
                # Make sure the imgs are converted to cuda tensor too
                imgs = imgs.cuda(async=True)
                
            imgsvar = torch.autograd.Variable(imgs)
            tgtsvar = torch.autograd.Variable(targets)

            # perform forward pass:
            out = model(imgsvar)
            loss = criterion(out, tgtsvar)

            # measure accuracy:
            prec1, prec5 = accuracy(out.data.cpu(), cpu_targets, topk=(1, 5))
            precs1.append(prec1[0] * targets.size(0))
            precs5.append(prec5[0] * targets.size(0))
            num_total += imgs.size(0)

            # compute gradient and do SGD step:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            bar.update(num_batches)

        # end-of-epoch hook:
        if end_epoch_hook is not None:
            prec1 = sum(precs1) / num_total
            prec5 = sum(precs5) / num_total
            end_epoch_hook(epoch, model, optimizer, prec1=prec1, prec5=prec5)

    # return trained model:
    return model


# helper function that test a model:
def _test(model, data_loader, return_probability=False):

    # assertions
    assert isinstance(model, torch.nn.Module)
    assert isinstance(data_loader, torch.utils.data.dataloader.DataLoader)

    # are we on CPU or GPU?
    is_gpu = not isinstance(model, torch.nn.backends.thnn.THNNFunctionBackend)
        
    # loop over data:
    model.eval()
    precs1, precs5, num_batches, num_total = [], [], 0, 0
    probs, all_targets = None, None
    bar = progressbar.ProgressBar(len(data_loader))
    bar.start()
    for num_batches, (imgs, targets) in enumerate(data_loader):

            # copy data to GPU:
        if is_gpu:
            cpu_targets = targets.clone()
            targets = targets.cuda(async=True)
            # Make sure the imgs are converted to cuda tensor too
            imgs = imgs.cuda(async=True)

        # perform prediction:
        imgsvar = torch.autograd.Variable(imgs.squeeze(), volatile=True)
        output = model(imgsvar)
        pred = output.data.cpu()

        if return_probability:
            probs = pred if probs is None else torch.cat((probs, pred), dim=0)
            all_targets = targets if all_targets is None else (
                torch.cat((all_targets, targets), dim=0))

        # measure accuracy:
        prec1, prec5 = accuracy(pred, cpu_targets, topk=(1, 5))
        precs1.append(prec1[0] * targets.size(0))
        precs5.append(prec5[0] * targets.size(0))
        num_total += imgs.size(0)
        bar.update(num_batches)

    if return_probability:
        return probs, all_targets
    else:
        # return average accuracy (@ 1 and 5):
        return sum(precs1) / num_total, sum(precs5) / num_total


def test(model, data_loader):
    return _test(model, data_loader, return_probability=False)


def get_prob(model, data_loader):
    return _test(model, data_loader, return_probability=True)
