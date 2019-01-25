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

import torch
import torch.nn as nn
import torch.optim
from lib.convnet import train, test
from lib.dataset import load_dataset, get_data_loader
from lib.defenses import get_defense
from lib.util import adjust_learning_rate, save_checkpoint
import lib.opts as opts
from lib.model import get_model


def _get_optimizer(model, args):
    if args.model.startswith('inceptionv4'):
        optimizer = torch.optim.RMSprop(
            model.parameters, lr=args.lr,
            alpha=args.rms_alpha, eps=args.rms_eps)
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), args.lr,
            momentum=args.momentum, weight_decay=args.weight_decay,
        )

    return optimizer


# run all the things:
def train_model(args):

    # At max 1 defense as no ensembling in training
    assert args.defenses is None or len(args.defenses) == 1
    defense_name = None if not args.defenses else args.defenses[0]
    defense = get_defense(defense_name, args)

    # Load model
    model, start_epoch, optimizer_ = get_model(
        args, load_checkpoint=args.resume, defense_name=defense_name, training=True)

    # set up optimizer:
    optimizer = _get_optimizer(model, args)

    # get from checkpoint if available
    if start_epoch and optimizer:
        args.start_epoch = start_epoch
        optimizer.load_state_dict(optimizer_)

    # set up criterion:
    criterion = nn.CrossEntropyLoss()
    
    if args.device == 'gpu':
        # Call .cuda() method on model
        criterion = criterion.cuda()
        model = model.cuda()

    loaders = {}

    # set up start-of-epoch hook:
    def start_epoch_hook(epoch, model, optimizer):
        print('| epoch %d, training:' % epoch)
        adjust_learning_rate(
            args.lr, epoch, optimizer,
            args.lr_decay, args.lr_decay_stepsize
        )

    # set up the end-of-epoch hook:
    def end_epoch_hook(epoch, model, optimizer, prec1=None, prec5=None):

        # print training error:
        if prec1 is not None:
            print('| training error @1 (epoch %d): %2.5f' % (epoch, 100. - prec1))
        if prec5 is not None:
            print('| training error @5 (epoch %d): %2.5f' % (epoch, 100. - prec5))

        # save checkpoint:
        print('| epoch %d, testing:' % epoch)
        save_checkpoint(args.models_root, {
            'epoch': epoch + 1,
            'model_name': args.model,
            'model_state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        })

        # measure validation error:
        prec1, prec5 = test(model, loaders['valid'])
        print('| validation error @1 (epoch %d: %2.5f' % (epoch, 100. - prec1))
        print('| validation error @5 (epoch %d: %2.5f' % (epoch, 100. - prec5))

    def data_loader_hook(epoch):
        # Reload data loader for epoch
        if args.preprocessed_epoch_data:
            print('| epoch %d, Loading data:' % epoch)
            for key in {'train', 'valid'}:
                # Load validation data only once
                if key == 'valid' and 'valid' in loaders:
                    break
                loaders[key] = get_data_loader(
                    load_dataset(args, key, defense, epoch=epoch),
                    batchsize=args.batchsize,
                    device=args.device,
                    shuffle=True,
                )
        # if data needs to be loaded only once and is not yet loaded
        elif len(loaders) == 0:
            print('| epoch %d, Loading data:' % epoch)
            for key in {'train', 'valid'}:
                loaders[key] = get_data_loader(
                    load_dataset(args, key, defense),
                    batchsize=args.batchsize,
                    device=args.device,
                    shuffle=True,
                )

        return loaders['train']

    # train the model:
    print('| training model...')
    train(model, criterion, optimizer,
          start_epoch_hook=start_epoch_hook,
          end_epoch_hook=end_epoch_hook,
          data_loader_hook=data_loader_hook,
          start_epoch=args.start_epoch,
          end_epoch=args.end_epoch,
          learning_rate=args.lr)
    print('| done.')


# run all the things:
if __name__ == '__main__':
    # parse input arguments:
    args = opts.parse_args(opts.OptType.TRAIN)
    train_model(args)
