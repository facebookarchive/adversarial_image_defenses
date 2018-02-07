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
from lib.constants import AdversaryType, DefenseType, AttackType


def _get_adv_path_part(args, defense_name, adv_params, with_defense=True):

    if args.adversary_to_generate:
        adversary = args.adversary_to_generate
    elif args.adversary:
        adversary = args.adversary
    else:
        return ""

    advtext = "adversary-"

    if args.attack_type == str(AttackType.WHITEBOX) and defense_name:
        advtext = advtext + defense_name + "_"

    if (adversary == str(AdversaryType.IFGS) or
            adversary == str(AdversaryType.DEEPFOOL)):
        assert adv_params is not None and adv_params['learning_rate'] is not None,\
            "learning rate can't be None for iterative attacks"
        advtext = '%s%s_%1.4f' % (advtext, adversary,
                                    adv_params['learning_rate'])
    else:
        advtext = advtext + adversary
        # non iterative attacks are generated at run time so for no defense
        if ((adversary == str(AdversaryType.CWL2) or
            adversary == str(AdversaryType.FGS)) and
            (adv_params is not None and adv_params['adv_strength'] is not None) and
                    defense_name and with_defense):
                advtext = '%s_%1.4f' % (advtext, adv_params['adv_strength'])

    return advtext


def _get_path_part_from_defenses(args, defense_name):
    assert 'preprocessed_data' in args, \
        'preprocessed_data argument is expected but not present in args'
    assert 'defenses' in args, \
        'defenses argument is expected but not present in args'
    assert 'quilting_random_stitch' in args, \
        'quilting_random_stitch argument is expected but not present in args'
    assert 'quilting_neighbors' in args, \
        'quilting_neighbors argument is expected but not present in args'
    assert 'tvm_weight' in args, \
        'tvm_weight argument is expected but not present in args'
    assert 'pixel_drop_rate' in args, \
        'pixel_drop_rate argument is expected but not present in args'

    if not defense_name:
        return ""

    d_str = "defense-" + defense_name
    if defense_name == str(DefenseType.TVM):
        d_str = str(d_str + '_drop_' + str(args.pixel_drop_rate) +
                    '_weight_' + str(args.tvm_weight))
    elif defense_name == str(DefenseType.QUILTING):
        if args.quilting_random_stitch:
            d_str = d_str + 'random-stitch'
        elif args.quilting_neighbors > 1:
            d_str = d_str + 'random-patch_' + str(args.quilting_neighbors)

    return d_str


def get_adversarial_file_path(args, root_dir, defense_name, adv_params, end_idx,
                                start_idx=0, data_batch_idx=None, with_defense=True):
    assert 'adversary' in args, \
        'partition argumenet is expected but not present in args'
    assert 'preprocessed_data' in args, \
        'preprocessed_data argumenet is expected but not present in args'
    assert 'adversary_model' in args, \
        'adversary_model argumenet is expected but not present in args'

    d_str = None
    if with_defense:
        d_str = _get_path_part_from_defenses(args, defense_name)
    adv_str = _get_adv_path_part(args, defense_name, adv_params,
                                    with_defense=with_defense)

    file_path = root_dir + '/'
    if d_str:
        file_path = file_path + d_str + '_'
    if adv_str:
        file_path = file_path + adv_str + '_'
    if args.adversary_model:
        file_path = file_path + args.adversary_model + '_'

    file_path = '%s%s_%d-%d' % (file_path, 'val', start_idx + 1, end_idx)

    if data_batch_idx is not None:
        file_path = str('%s_%d' % (file_path, data_batch_idx))
    file_path = file_path + '.pth'

    return file_path


def _get_preprocessed_tar_index_dir(args, data_type, epoch):
    assert (data_type == 'train' or data_type == 'valid'), (
        "{} data type not defined. Defined types are \"train\" "
        "and \"valid\" ".format(data_type))
    assert os.path.isdir(args.tar_index_dir), \
        "{} doesn't exist".format(args.tar_index_dir)
    # preprocessed train dataset for all epochs
    if data_type == 'train':
        assert epoch >= 0
        index_file = "{}/epoch_{}.index".format(args.tar_index_dir, epoch)
    # For validation, same dataset for all epochs
    else:
        index_file = "{}/val.index".format(args.tar_index_dir)
    assert os.path.isfile(index_file), \
        "{} doesn't exist".format(index_file)

    return index_file


# get location of the images
def get_img_dir(args,
                data_type,
                epoch=-1):

    img_dir = None
    # Images are not adversarial
    if 'adversary' not in args or not args.adversary:
        if 'preprocessed_data' not in args or not args.preprocessed_data:
            dir_name = 'train' if data_type == 'train' else 'val'
            img_dir = str(os.path.join(args.imagenet_dir, dir_name))
        # Data is preprocessed for defenses like tvm, quilting
        else:
            # this is index_file stored as img_dir
            img_dir = str(_get_preprocessed_tar_index_dir(args, data_type, epoch))

    # If needs to work on pre-generated adversarial images
    else:
        img_dir = str(args.adversarial_root)

    assert os.path.isdir(img_dir) or os.path.isfile(img_dir), \
        "Data directory {} doesn't exist. Update the IMAGENET_DIR1 in the \
            config file with correct path".format(img_dir)

    return img_dir


def get_quilting_filepaths(args):
    root = args.quilting_patch_root
    size = args.quilting_patch_size
    patches_filename = str("{root}/patches_{size}.pickle".format(
        root=root, size=size))
    index_filename = str("{root}/index_{size}.faiss".format(
        root=root, size=size))

    return patches_filename, index_filename
