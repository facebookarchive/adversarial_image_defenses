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
import lib.opts as opts
from lib.dataset import load_dataset, get_data_loader
import lib.adversary as adversary
from lib.model import get_model
import lib.constants as constants
from lib.constants import AdversaryType
from lib.paths import get_adversarial_file_path
from lib.transformations.transforms import Unnormalize, Normalize
import os
from enum import Enum


class OperationType(Enum):
    GENERATE_ADVERSARIAL = 'generate_adversarial'
    CONCAT_ADVERSARIAL = 'concat_adversarial'
    COMPUTE_STATS = 'compute_adversarial_stats'

    @classmethod
    def has_value(cls, value):
        return (any(value == item.value for item in cls))

    def __str__(self):
        return str(self.value)


def _get_data_indices(args):
    assert 'partition' in args, \
        'partition argumenet is expected but not present in args'
    assert 'partition_size' in args, \
        'partition_size argumenet is expected but not present in args'

    data_indices = {}
    data_indices['start_idx'] = args.partition * args.partition_size
    data_indices['end_idx'] = (args.partition + 1) * args.partition_size
    return data_indices


# Concat adversarial data generated from batches
def concat_adversarial(args):
    assert not args.partition_size == 0, \
        "partition_size can't be zero"
    assert 'learning_rate' in args, \
        "adv_params are not provided"
    assert len(args.learning_rate) == 1, \
        "adv_params are not provided"

    defense_name = None if not args.defenses else args.defenses[0]
    adv_params = {
        'learning_rate': args.learning_rate[0],
        'adv_strength': None
    }

    end_idx = args.n_samples
    nfiles = end_idx // args.partition_size
    for i in range(nfiles):
        start_idx = (i * args.partition_size) + 1
        partition_end = (i + 1) * args.partition_size
        partition_file = get_adversarial_file_path(
            args, args.adversarial_root, defense_name, adv_params, partition_end,
            start_idx, with_defense=False)

        assert os.path.isfile(partition_file), \
            "No file found at " + partition_file
        print('| Reading file ' + partition_file)
        result = torch.load(partition_file)
        inputs = result['all_inputs']
        outputs = result['all_outputs']
        targets = result['all_targets']
        status = result['status']
        targets = torch.LongTensor(targets)
        if i == 0:
            all_inputs = inputs
            all_outputs = outputs
            all_targets = targets
            all_status = status
        else:
            all_inputs = torch.cat((all_inputs, inputs), 0)
            all_outputs = torch.cat((all_outputs, outputs), 0)
            all_status = torch.cat((all_status, status), 0)
            all_targets = torch.cat((all_targets, targets), 0)
        # print(all_inputs.size())

    out_file = get_adversarial_file_path(args, args.adversarial_root,
                                                defense_name, adv_params,
                                                nfiles * args.partition_size,
                                                args.partition + 1,
                                                with_defense=False)

    if not os.path.isdir(args.adversarial_root):
        os.mkdir(args.adversarial_root)
    print('| Writing concatenated adversarial data to ' + out_file)
    torch.save({'status': all_status, 'all_inputs': all_inputs,
                'all_outputs': all_outputs, 'all_targets': all_targets},
                out_file)


def compute_stats(args):
    assert not args.partition_size == 0, \
        "partition_size can't be zero"
    assert 'learning_rate' in args and 'adv_strength' in args, \
        "adv_params are not provided"

    defense_name = None if not args.defenses else args.defenses[0]
    adv_params = constants.get_adv_params(args)
    print('| adv_params:', adv_params)
    start_idx = 0
    end_idx = args.n_samples
    in_file = get_adversarial_file_path(
        args, args.adversarial_root, defense_name, adv_params, end_idx,
        start_idx, with_defense=True)
    assert os.path.isfile(in_file), \
        "No file found at " + in_file
    print('| Reading file ' + in_file)
    result = torch.load(in_file)
    all_inputs = result['all_inputs']
    all_outputs = result['all_outputs']

    normalize = Normalize(args.data_params['MEAN_STD']['MEAN'],
                            args.data_params['MEAN_STD']['STD'])
    all_inputs = normalize(all_inputs)
    all_outputs = normalize(all_outputs)
    rb, _ssim, sc = adversary.compute_stats(
        all_inputs, all_outputs, result['status'])
    print('average robustness = ' + str(rb))
    print('success rate = ' + str(sc))


def generate_adversarial_images(args):
    # assertions
    assert args.adversary_to_generate is not None, \
        "adversary_to_generate can't be None"
    assert AdversaryType.has_value(args.adversary_to_generate), \
        "\"{}\" adversary_to_generate not defined".format(args.adversary_to_generate)

    defense_name = None if not args.defenses else args.defenses[0]
    # defense = get_defense(defense_name, args)
    data_indices = _get_data_indices(args)
    data_type = args.data_type if args.data_type == "train" else "valid"
    dataset = load_dataset(args, data_type, None, data_indices=data_indices)
    data_loader = get_data_loader(
        dataset,
        batchsize=args.batchsize,
        device=args.device,
        shuffle=False)

    model, _, _ = get_model(args, load_checkpoint=True, defense_name=defense_name)

    adv_params = constants.get_adv_params(args)
    print('| adv_params:', adv_params)
    status = None
    all_inputs = None
    all_outputs = None
    all_targets = None
    bar = progressbar.ProgressBar(len(data_loader))
    bar.start()
    for batch_num, (imgs, targets) in enumerate(data_loader):
        if args.adversary_to_generate == str(AdversaryType.DEEPFOOL):
            assert adv_params['learning_rate'] is not None
            s, r = adversary.deepfool(
                model, imgs, targets, args.data_params['NUM_CLASSES'],
                train_mode=(args.data_type == 'train'), max_iter=args.max_adv_iter,
                step_size=adv_params['learning_rate'], batch_size=args.batchsize,
                labels=dataset.get_classes())
        elif args.adversary_to_generate == str(AdversaryType.FGS):
            s, r = adversary.fgs(
                model, imgs, targets, train_mode=(args.data_type == 'train'),
                mode=args.fgs_mode)
        elif args.adversary_to_generate == str(AdversaryType.IFGS):
            assert adv_params['learning_rate'] is not None
            s, r = adversary.ifgs(
                model, imgs, targets,
                train_mode=(args.data_type == 'train'), max_iter=args.max_adv_iter,
                step_size=adv_params['learning_rate'], mode=args.fgs_mode)
        elif args.adversary_to_generate == str(AdversaryType.CWL2):
            assert args.adv_strength is not None and len(args.adv_strength) == 1
            if len(args.crop_frac) == 1:
                crop_frac = args.crop_frac[0]
            else:
                crop_frac = 1.0
            s, r = adversary.cw(
                model, imgs, targets, args.adv_strength[0], 'l2',
                tv_weight=args.tvm_weight,
                train_mode=(args.data_type == 'train'), max_iter=args.max_adv_iter,
                drop_rate=args.pixel_drop_rate, crop_frac=crop_frac,
                kappa=args.margin)
        elif args.adversary_to_generate == str(AdversaryType.CWLINF):
            assert args.adv_strength is not None and len(args.adv_strength) == 1
            s, r = adversary.cw(
                model, imgs, targets, args.adv_strength[0], 'linf',
                bound=args.adv_bound,
                tv_weight=args.tvm_weight,
                train_mode=(args.data_type == 'train'), max_iter=args.max_adv_iter,
                drop_rate=args.pixel_drop_rate, crop_frac=args.crop_frac,
                kappa=args.margin)

        if status is None:
            status = s.clone()
            all_inputs = imgs.clone()
            all_outputs = imgs + r
            all_targets = targets.clone()
        else:
            status = torch.cat((status, s), 0)
            all_inputs = torch.cat((all_inputs, imgs), 0)
            all_outputs = torch.cat((all_outputs, imgs + r), 0)
            all_targets = torch.cat((all_targets, targets), 0)
        bar.update(batch_num)

    print("| computing adversarial stats...")
    if args.compute_stats:
        rb, ssim, sc = adversary.compute_stats(all_inputs, all_outputs, status)
        print('| average robustness = ' + str(rb))
        print('| average SSIM = ' + str(ssim))
        print('| success rate = ' + str(sc))

    # Unnormalize before saving
    unnormalize = Unnormalize(args.data_params['MEAN_STD']['MEAN'],
                                args.data_params['MEAN_STD']['STD'])
    all_inputs = unnormalize(all_inputs)
    all_outputs = unnormalize(all_outputs)
    # save output
    output_file = get_adversarial_file_path(
        args, args.adversarial_root, defense_name, adv_params,
        data_indices['end_idx'], start_idx=data_indices['start_idx'],
        with_defense=False)
    print("| Saving adversarial data at " + output_file)
    if not os.path.isdir(args.adversarial_root):
        os.makedirs(args.adversarial_root)
    torch.save({'status': status, 'all_inputs': all_inputs,
                'all_outputs': all_outputs, 'all_targets': all_targets},
                output_file)


def main():
    # parse input arguments:
    args = opts.parse_args(opts.OptType.ADVERSARIAL)

    # Only runs one method at a time
    assert args.operation is not None, \
        "operation to run can't be None"
    assert OperationType.has_value(args.operation), \
        "\"{}\" operation not defined".format(args.operation)

    if args.operation == str(OperationType.GENERATE_ADVERSARIAL):
        generate_adversarial_images(args)
    elif args.operation == str(OperationType.CONCAT_ADVERSARIAL):
        concat_adversarial(args)
    elif args.operation == str(OperationType.COMPUTE_STATS):
        compute_stats(args)


# run:
if __name__ == '__main__':
    main()
