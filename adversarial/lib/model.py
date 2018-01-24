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
import pkgutil
import torch.nn as nn
import torch.nn.parallel
import torchvision.models as models
from lib.util import load_checkpoint
import lib.constants as constants


def _load_torchvision_model(model, pretrained=True):
    assert hasattr(models, model), (
        "Model {} is not available in torchvision.models."
        "Supported models are: {}".format(model, constants.MODELS))
    model = getattr(models, model)(pretrained=pretrained)
    return model


def _init_data_parallel(model, device):
    if device == 'gpu':
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True
        model = torch.nn.DataParallel(model).cuda()
        return model


def _init_inceptionresnetv2(model, device):
    # inceptionresnetv2 has default 1001 classes,
    # get rid of first background class
    new_classif = nn.Linear(1536, 1000)
    new_classif.weight.data = model.classif.weight.data[1:]
    new_classif.bias.data = model.classif.bias.data[1:]
    model.classif = new_classif
    model = _init_data_parallel(model, device)
    return model


def _load_model_from_checkpoint(model, model_path, model_name,
                                defense_name=None, training=False):

    assert model is not None, "Model should not be None"
    assert model_path and model_name, \
        "Model path is not provided"
    model_path = os.path.join(model_path, model_name)
    if defense_name is not None:
        model_path = str(model_path + '_' + defense_name)

    if not training:
        assert os.path.isdir(model_path), \
            "Model directory doesn't exist at: {}".format(model_path)

    print('| loading model from checkpoint %s' % model_path)
    checkpoint = load_checkpoint(model_path)
    start_epoch, optimizer = None, None
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
        if training:
            start_epoch = checkpoint['epoch']
            optimizer = checkpoint['optimizer']
    else:
        print('| no model available at %s...' % model_path)

    return model, start_epoch, optimizer


def get_model(args, load_checkpoint=False, defense_name=None, training=False):

    assert (args.model in constants.MODELS), ("%s not a supported model" % args.model)
    model, start_epoch, optimizer = None, None, None
    # load model:
    print('| loading model...')
    if args.model == 'inception_v4':
        assert "NUM_CLASSES" in args.data_params, \
            "Inception parameters should have number of classes defined"
        if args.pretrained:
            pretrained = 'imagenet'
        else:
            pretrained = None
        assert pkgutil.find_loader("lib.models.inceptionv4") is not None, \
            ("Module lib.models.inceptionv4 can't be found. "
             "Check the setup script and rebuild again to download")
        from lib.models.inceptionv4 import inceptionv4
        model = inceptionv4(pretrained=pretrained)
    elif args.model == 'inceptionresnetv2':
        assert not args.pretrained, \
            "For inceptionresnetv2 pretrained not available"
        assert pkgutil.find_loader("lib.models.inceptionresnetv2") is not None, \
            ("Module lib.models.inceptionresnetv2 can't be found. "
             "Check the setup script and rebuild again to download")
        from lib.models.inceptionresnetv2 import InceptionResnetV2
        model = InceptionResnetV2()
    else:
        model = _load_torchvision_model(args.model,
                                        pretrained=args.pretrained)

    # inceptionresnetv2 from adversarial ensemble training at checkpoint
    # is not saved with DataParallel
    # TODO: Save it with DataParallel to cleanup code below
    if not args.model == 'inceptionresnetv2':
        model = _init_data_parallel(model, args.device)

    if load_checkpoint and not args.pretrained:
        model, start_epoch, optimizer = _load_model_from_checkpoint(
            model, args.models_root, args.model, defense_name,
            training=training)

    if args.model == 'inceptionresnetv2':
        model = _init_inceptionresnetv2(model, args.device)

    return model, start_epoch, optimizer
