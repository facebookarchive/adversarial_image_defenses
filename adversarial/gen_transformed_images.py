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

from lib.dataset import load_dataset
from lib.defenses import get_defense
import torch
import torchvision.transforms as trans
import os
import lib.paths as paths
import lib.opts as opts
from enum import Enum
import lib.constants as constants
from multiprocessing.pool import ThreadPool
from torchvision.transforms import ToPILImage


class OperationType(Enum):
    TRANSFORM_ADVERSARIAL = 'transformation_on_adv'
    TRANSFORM_RAW = 'transformation_on_raw'
    CAT_DATA = 'concatenate_data'
    SAVE_SAMPLES = 'save_samples'

    @classmethod
    def has_value(cls, value):
        return (any(value == item.value for item in cls))

    def __str__(self):
        return str(self.value)


def _get_start_end_index(args):
    assert 'partition' in args, \
        'partition argumenet is expected but not present in args'
    assert 'partition_size' in args, \
        'partition_size argumenet is expected but not present in args'

    start_idx = args.partition * args.partition_size
    end_idx = (args.partition + 1) * args.partition_size
    return start_idx, end_idx


def _get_out_file(output_root, defense_name, target, file_name):
    output_dir = '{root}/{transfomation}/{target}'.format(
        root=output_root, transfomation=defense_name, target=target)
    if not os.path.isdir(output_dir):
        try:
            os.makedirs(output_dir)
        except OSError as exception:
            import errno
            if exception.errno != errno.EEXIST:
                raise
    output_filepath = '{path}/{fname}'.format(path=output_dir, fname=file_name)
    return output_filepath


# setup partial dataset
def _load_partial_dataset(args, data_type, defense, adv_params):
    start_idx, end_idx = _get_start_end_index(args)
    data_indices = {'start_idx': start_idx, 'end_idx': end_idx}
    dataset = load_dataset(args, data_type, defense, adv_params,
                            data_indices=data_indices)
    return dataset


# Concat data generated from batches
def concatenate_data(args, defense_name, adv_params, data_batch_idx=None):
    assert not args.partition_size == 0, \
        "partition_size can't be zero"

    end_idx = args.n_samples
    nfiles = end_idx // args.partition_size
    for i in range(nfiles):
        start_idx = (i * args.partition_size) + 1
        partition_end = (i + 1) * args.partition_size
        partition_file = paths.get_adversarial_file_path(
            args, args.partition_dir, defense_name, adv_params, partition_end,
            start_idx, data_batch_idx)

        assert os.path.isfile(partition_file), \
            "No file found at " + partition_file
        print('| Reading file ' + partition_file)
        result = torch.load(partition_file)
        inputs = result['all_outputs']
        targets = result['all_targets']
        targets = torch.LongTensor(targets)
        if i == 0:
            all_imgs = inputs
            all_targets = targets
        else:
            all_imgs = torch.cat((all_imgs, inputs), 0)
            all_targets = torch.cat((all_targets, targets), 0)

    out_file = paths.get_adversarial_file_path(args, args.out_dir,
                                                defense_name, adv_params,
                                                nfiles * args.partition_size,
                                                args.partition + 1)

    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)
    print('| Writing concatenated data to ' + out_file)
    torch.save({'all_outputs': all_imgs, 'all_targets': all_targets},
               out_file)


# Apply transformations on adversarial images
def transformation_on_adv(args, dataset, defense_name, adv_params,
                            data_batch_idx=None):
    pool = ThreadPool(args.n_threads)

    def generate(idx):
        return dataset[idx]

    dataset = pool.map(generate, range(len(dataset)))
    pool.close()
    pool.join()

    # save all data in a file
    all_adv = []
    all_targets = []
    for item in dataset:
        all_adv.append(item[0])
        all_targets.append(item[1])

    all_adv = torch.stack(all_adv, 0)
    all_targets = torch.LongTensor(all_targets)
    if not os.path.isdir(args.partition_dir):
        os.makedir(args.partition_dir)
    start_idx, end_idx = _get_start_end_index(args)
    out_file = paths.get_adversarial_file_path(
        args, args.partition_dir, defense_name, adv_params,
        end_idx, start_idx, data_batch_idx)
    torch.save({'all_outputs': all_adv,
                'all_targets': all_targets},
               out_file)

    print('Saved Transformed tensor at ' + out_file)
    dataset = None
    all_adv = None


def transformation_on_raw(args, dataset, defense_name):
    pool = ThreadPool(args.n_threads)
    if not os.path.isdir(args.partition_dir):
        os.makedirs(args.partition_dir)

    def generate(idx):
        img, target_index, file_name = dataset[idx]
        target = dataset.get_idx_to_class(target_index)
        out_file = _get_out_file(args.partition_dir, defense_name, target, file_name)
        ToPILImage()(img).save(out_file)

    pool.map(generate, range(len(dataset)))


def save_samples(args):
    assert args.data_file is not None and os.path.isfile(args.data_file), \
        "Data file path required"

    basename = os.path.basename(args.data_file)
    # Validate if generated data is good
    result = torch.load(args.data_file)
    outputs = result['all_outputs']
    for i in range(10):
        img = trans.ToPILImage()(outputs[i])
        img_path = str("/tmp/test_img_" + basename + "_" + str(i) + ".JPEG")
        print("saving image: " + img_path)
        img.save(img_path)


def generate_transformed_images(args):

    # Only runs one method at a time
    assert args.operation is not None, \
        "operation to run can't be None"
    assert OperationType.has_value(args.operation), \
        "\"{}\" operation not defined".format(args.operation)

    assert args.defenses is not None, "Defenses can't be None"
    assert not args.preprocessed_data, \
        "Trying to apply transformations on already transformed images"

    if args.operation == str(OperationType.TRANSFORM_ADVERSARIAL):
        for idx, defense_name in enumerate(args.defenses):
            defense = get_defense(defense_name, args)
            adv_params = constants.get_adv_params(args, idx)
            print("| adv_params: ", adv_params)
            dataset = _load_partial_dataset(args, 'valid', defense, adv_params)

            if args.data_batches is None:
                transformation_on_adv(args, dataset, defense_name, adv_params)
            else:
                for i in range(args.data_batches):
                    transformation_on_adv(args, dataset, defense_name, adv_params,
                                            data_batch_idx=i)

    elif args.operation == str(OperationType.CAT_DATA):
        for idx, defense_name in enumerate(args.defenses):
            adv_params = constants.get_adv_params(args, idx)
            print("| adv_params: ", adv_params)
            if args.data_batches is None:
                concatenate_data(args, defense_name, adv_params)
            else:
                for i in range(args.data_batches):
                    concatenate_data(args, defense_name, adv_params, data_batch_idx=i)

    elif args.operation == str(OperationType.TRANSFORM_RAW):
        start_class_idx = args.partition * args.partition_size
        end_class_idx = (args.partition + 1) * args.partition_size
        class_indices = range(start_class_idx, end_class_idx)
        for defense_name in args.defenses:
            defense = get_defense(defense_name, args)
            data_type = args.data_type if args.data_type == "train" else "valid"
            dataset = load_dataset(args, data_type, defense,
                                    class_indices=class_indices)
            transformation_on_raw(args, dataset, defense_name)


# run:
if __name__ == '__main__':
    # parse input arguments:
    args = opts.parse_args(opts.OptType.TRANSFORMATION)
    generate_transformed_images(args)
