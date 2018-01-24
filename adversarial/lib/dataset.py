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
from lib.datasets.sub_dataset_folder import PartialImageFolder
from lib.datasets.sub_dataset_tarfolder import PartialTarFolder
from lib.datasets.transform_dataset import TransformDataset
from lib.transformations.transformation_helper import setup_transformations
import lib.paths as paths
import os
from lib.datasets.dataset_classes_folder import ImageClassFolder
from six import string_types


# helper function for loading adversarial data:
def _load_adversarial_helper(path, eps=0,
                                normalize=True, mask=None, preprocessed=False):

    # assertions:
    assert isinstance(path, string_types)
    if mask is not None:
        assert torch.is_tensor(mask)

    # load file with images and perturbations:
    result = torch.load(path)
    all_outputs = result['all_outputs']
    all_targets = result['all_targets']

    if not preprocessed:
        # construct adversarial examples:
        if eps > 0:
            all_inputs = result['all_inputs']
            r = all_outputs - all_inputs
            if normalize:
                r = r.sign()
            all_outputs = all_inputs + eps * r
        if mask is not None:
            all_idx = torch.arange(0, all_outputs.size(0)).long()
            mask_idx = all_idx[mask]
            all_outputs = all_outputs[mask_idx]
            all_targets = all_targets[mask_idx]

    return all_outputs, all_targets


# function that loads data for a given adversary
def load_adversarial(args, img_dir, defense_name, adv_params,
                     data_batch_idx=None):
    # assertions:
    assert isinstance(img_dir, string_types)

    data_file = paths.get_adversarial_file_path(
        args, img_dir, defense_name, adv_params, args.n_samples,
        data_batch_idx=data_batch_idx,
        with_defense=args.preprocessed_data)

    assert os.path.isfile(data_file), \
        "No file found at " + data_file
    # load the adversarial examples"
    print('| loading adversarial examples from %s...' % data_file)
    if not (args.adversary == 'cwl2' or args.adversary == 'fgs'):
        adv_strength = 0.0
    else:
        adv_strength = adv_params['adv_strength']

    normalize = True
    if args.adversary == 'cwl2' and adv_strength > 0:
        normalize = False
    return _load_adversarial_helper(data_file, adv_strength,
                                      normalize=normalize,
                                      preprocessed=args.preprocessed_data)


# init dataset
def get_dataset(args, img_dir, defense_name, adv_params, transform,
                data_batch_idx=None, class_indices=None, data_indices=None):

    # assertions
    if 'preprocessed_data' in args and args.preprocessed_data:
        assert defense_name is not None, (
            "If data is already pre processed for defenses then "
            "defenses can't be None")

    # for data without adversary
    if 'adversary' not in args or args.adversary is None:
        # For pre-applied defense, read data from tar files
        if 'preprocessed_data' in args and args.preprocessed_data:
            # get prefix in tar member names
            if defense_name:
                if args.tar_prefix:
                    tar_prefix = str(args.tar_prefix + '/' + defense_name)
                else:
                    tar_prefix = defense_name
            else:
                tar_prefix = args.tar_prefix
            dataset = PartialTarFolder(img_dir, path_prefix=tar_prefix,
                                        transform=transform)
        else:
            if class_indices:
                # Load data for only target classes(helpful in parallel processing)
                dataset = ImageClassFolder(img_dir, class_indices, transform)
            else:
                # dataset = ImageFolder(
                #     img_dir, transform=transform)
                dataset = PartialImageFolder(
                    img_dir, data_indices=data_indices, transform=transform)

    else:  # adversary
        # Load adversarial dataset
        adv_data, targets = load_adversarial(args, img_dir,
                                                defense_name, adv_params,
                                                data_batch_idx=data_batch_idx)
        dataset = TransformDataset(
            torch.utils.data.TensorDataset(adv_data, targets), transform, data_indices)

    return dataset


def load_dataset(args, data_type, defense, adv_params=None, crop=None,
                    epoch=-1, data_batch_idx=None,
                    class_indices=None, data_indices=None,
                    with_transformation=True):

    # assertions:
    assert (data_type == 'train' or data_type == 'valid'), (
        "{} data type not defined. Defined types are \"train\" "
        "and \"valid\" ".format(data_type))
    if defense is not None:
        assert callable(defense), (
            "defense should be a callable method")

    # get data directory
    img_dir = paths.get_img_dir(args, data_type, epoch=epoch)

    # setup transformations to apply on loaded data
    transform = None
    if with_transformation:
        transform = setup_transformations(args, data_type, defense,
                                            crop=crop)
    defense_name = None if defense is None else defense.get_name()
    # initialize dataset
    print('| Loading data from ' + img_dir)
    dataset = get_dataset(args, img_dir, defense_name, adv_params, transform,
                            data_batch_idx=data_batch_idx,
                            class_indices=class_indices,
                            data_indices=data_indices)
    return dataset


# function that constructs a data loader for a dataset:
def get_data_loader(dataset, batchsize=32, workers=10, device='cpu', shuffle=True):

    # assertions:
    assert isinstance(dataset, torch.utils.data.dataset.Dataset)
    assert type(batchsize) == int and batchsize > 0
    assert type(workers) == int and workers >= 0
    assert device == 'cpu' or device == 'gpu'

    # construct data loader:
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batchsize,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=(device == 'gpu'),
    )


# helper function to make image look pretty:
def visualize_image(args, img, normalize=True):
    assert torch.is_tensor(img) and img.dim() == 3
    new_img = img.clone()
    if normalize:
        for c in range(new_img.size(0)):
            new_img[c].mul_(
                args.data_params['MEAN_STD']['STD'][c]).add_(
                    args.data_params['MEAN_STD']['MEAN'][c])
    return torch.mul(new_img, 255.).byte()
