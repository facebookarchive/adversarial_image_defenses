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
import torch.nn.parallel
import torch.optim
import torch.utils.data

from lib.convnet import get_prob
from lib.dataset import load_dataset, get_data_loader
from lib.defenses import get_defense
import lib.opts as opts
from lib.model import get_model
import lib.transformations.transforms as transforms
from lib.transformations.transformation_helper import update_dataset_transformation
import lib.constants as constants
from lib.constants import DefenseType
from lib.util import accuracy

ENSEMBLE_TYPE = ['max', 'avg']


# Test and ensemble image crops
def _eval_crops(args, dataset, model, defense, crop, ncrops, crop_type):

    # assertions
    assert dataset is not None, "dataset expected"
    assert model is not None, "model expected"
    assert crop_type is None or isinstance(crop_type, str)
    if crop is not None:
        assert callable(crop)
    assert type(ncrops) == int

    probs = None

    for crop_num in range(ncrops):

        # For sliding crop update crop function in dataset
        if crop_type == 'sliding':
            crop.update_sliding_position(crop_num)
            dataset = update_dataset_transformation(
                dataset, args, 'valid', defense, crop)

        # set up dataloader:
        print('| set up data loader...')
        data_loader = get_data_loader(
            dataset,
            batchsize=args.batchsize,
            device=args.device,
            shuffle=False,
        )

        # test
        prob, targets = get_prob(model, data_loader)
        # collect prob for each run
        if probs is None:
            probs = torch.zeros(ncrops, len(dataset), prob.size(1))
        probs[crop_num, :, :] = prob

        # measure and print accuracy
        _, _prob = prob.topk(5, dim=1)
        _correct = _prob.eq(targets.view(-1, 1).expand_as(_prob))
        _top1 = _correct.select(1, 0).float().mean() * 100
        defense_name = "no defense" if defense is None else defense.get_name()
        print('| crop[%d]: top1 acc for %s = %f' % (crop_num, defense_name, _top1))

        data_loader = None

    return probs, targets


def classify_images(args):

    # assertions
    assert args.ensemble is None or args.ensemble in ENSEMBLE_TYPE, \
        "{} not a supported type. Only supported ensembling are {}".format(
            args.ensemble, ENSEMBLE_TYPE)
    if not args.ensemble:
        assert args.ncrops is None or (
            len(args.ncrops) == 1 and args.ncrops[0] == 1)
    if args.defenses is not None:
        for d in args.defenses:
            assert DefenseType.has_value(d), \
                "\"{}\" defense not defined".format(d)
        # crops expected for each defense
        assert (args.ncrops is None or
                len(args.ncrops) == len(args.defenses)), (
            "Number of crops for each defense is expected")
        assert (args.crop_type is None or
                len(args.crop_type) == len(args.defenses)), (
            "crop_type for each defense is expected")
        # assert (len(args.crop_frac) == len(args.defenses)), (
        #     "crop_frac for each defense is expected")
    elif args.ncrops is not None:
        # no crop ensembling when defense is None
        assert len(args.ncrops) == 1
        assert args.crop_frac is not None and len(args.crop_frac) == 1, \
            "Only one crop_frac is expected as there is no defense"
        assert args.crop_type is not None and len(args.crop_type) == 1, \
            "Only one crop_type is expected as there is no defense"

    if args.defenses is None or len(args.defenses) == 0:
        defenses = [None]
    else:
        defenses = args.defenses

    all_defense_probs = None
    for idx, defense_name in enumerate(defenses):
        # initialize dataset
        defense = get_defense(defense_name, args)
        # Read preset params for adversary based on args
        adv_params = constants.get_adv_params(args, idx)
        print("| adv_params: ", adv_params)
        # setup crop
        ncrops = 1
        crop_type = None
        crop_frac = 1.0
        if args.ncrops:
            crop_type = args.crop_type[idx]
            crop_frac = args.crop_frac[idx]
            if crop_type == 'sliding':
                ncrops = 9
            else:
                ncrops = args.ncrops[idx]
        # Init custom crop function
        crop = transforms.Crop(crop_type, crop_frac)
        # initialize dataset
        dataset = load_dataset(args, 'valid', defense, adv_params, crop)
        # load model
        model, _, _ = get_model(args, load_checkpoint=True, defense_name=defense_name)

        # get crop probabilities for crops for current defense
        probs, targets = _eval_crops(args, dataset, model, defense,
                                        crop, ncrops, crop_type)

        if all_defense_probs is None:
            all_defense_probs = torch.zeros(len(defenses),
                                            len(dataset),
                                            probs.size(2))
        # Ensemble crop probabilities
        if args.ensemble == 'max':
            probs = torch.max(probs, dim=0)[0]
        elif args.ensemble == 'avg':  # for average ensembling
            probs = torch.mean(probs, dim=0)
        else:  # for no ensembling
            assert all_defense_probs.size(0) == 1
            probs = probs[0]
        all_defense_probs[idx, :, :] = probs

        # free memory
        dataset = None
        model = None

    # Ensemble defense probabilities
    if args.ensemble == 'max':
        all_defense_probs = torch.max(all_defense_probs, dim=0)[0]
    elif args.ensemble == 'avg':  # for average ensembling
        all_defense_probs = torch.mean(all_defense_probs, dim=0)
    else:  # for no ensembling
        assert all_defense_probs.size(0) == 1
        all_defense_probs = all_defense_probs[0]
    # Calculate top1 and top5 accuracy
    prec1, prec5 = accuracy(all_defense_probs, targets, topk=(1, 5))
    print('=' * 50)
    print('Results for model={}, attack={}, ensemble_type={} '.format(
        args.model, args.adversary, args.ensemble))
    prec1 = prec1[0]
    prec5 = prec5[0]
    print('| classification accuracy @1: %2.5f' % (prec1))
    print('| classification accuracy @5: %2.5f' % (prec5))
    print('| classification error @1: %2.5f' % (100. - prec1))
    print('| classification error @5: %2.5f' % (100. - prec5))
    print('| done.')


# run:
if __name__ == '__main__':
    # parse input arguments
    args = opts.parse_args(opts.OptType.CLASSIFY)
    classify_images(args)
