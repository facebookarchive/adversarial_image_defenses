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

import argparse
import os
from enum import Enum
import lib.constants as constants
import json


# Init paths from config
path_config_file = str("path_config.json")
if not os.path.isfile(path_config_file):
    path_config_file = os.path.join(os.path.dirname(__file__), str("path_config.json"))
assert os.path.isfile(path_config_file), \
    "path_config.json file not found at {}".format(path_config_file)
path_config = json.load(open(path_config_file, "r"))

DATA_ROOT = path_config["DATA_ROOT"]
QUILTING_ROOT = path_config["QUILTING_ROOT"]
MODELS_ROOT = path_config["MODELS_ROOT"]
if not path_config["IMAGENET_DIR1"]:
    IMAGENET_DIR1 = os.path.join(os.path.dirname(__file__), "../test/images")
else:
    IMAGENET_DIR1 = path_config["IMAGENET_DIR1"]
if not path_config["IMAGENET_DIR2"]:
    IMAGENET_DIR2 = None
else:
    IMAGENET_DIR2 = path_config["IMAGENET_DIR2"]

# path to save/load tarred transformed data
TAR_DIR = DATA_ROOT + '/imagenet_transformed_tarred'
# path to save/load index objects for tarred data
# used to directly reads tar data without untarring fully(much faster)
TAR_INDEX_DIR = DATA_ROOT + '/imagenet_transformed_tarred_index'


class OptType(Enum):
    QUILTING_PATCHES = 'QUILTING_PATCHES'
    TRAIN = 'TRAIN'
    CLASSIFY = 'CLASSIFY'
    TRANSFORMATION = 'TRANSFORMATION'
    ADVERSARIAL = 'ADVERSARIAL'


def _setup_common_args(parser):
    # paths
    parser.add_argument('--data_root', default=DATA_ROOT, type=str, metavar='N',
                        help='Main data directory to save and read data')
    parser.add_argument('--models_root', default=MODELS_ROOT, type=str, metavar='N',
                        help='Directory to store/load models')
    parser.add_argument('--tar_dir', default=TAR_DIR, type=str, metavar='N',
                        help='Path for directory with processed(transformed)'
                                'tar train/val files')
    parser.add_argument('--tar_index_dir', default=TAR_INDEX_DIR, type=str, metavar='N',
                        help='Path for directory with processed tar index files')
    parser.add_argument('--tar_prefix', default='/tmp/imagenet_transformed',
                        type=str, metavar='N',
                        help='Path prefix in all tar files')
    parser.add_argument('--quilting_index_root', default=QUILTING_ROOT,
                        type=str, metavar='N',
                        help='Path for quilting index files')
    parser.add_argument('--quilting_patch_root', default=QUILTING_ROOT,
                        type=str, metavar='N',
                        help='the path for quilting patches')

    parser.add_argument('--model', default='resnet50', type=str, metavar='N',
                        help='model to use (default: resnet50)')
    parser.add_argument('--device', default='gpu', type=str, metavar='N',
                        help='device to use: cpu or gpu (default = gpu)')
    # Set normalize to True for training, testing and generating adversarial images.
    # For generating transformations, let it be False.
    parser.add_argument('--normalize', default=False, action='store_true',
                        help='Normalize image data.')
    parser.add_argument('--batchsize', default=256, type=int, metavar='N',
                        help='batch size (default = 256)')
    parser.add_argument('--preprocessed_data', default=False, action='store_true',
                        help='Defenses are already applied on saved images')
    parser.add_argument('--defenses', default=None, nargs='*', type=str, metavar='N',
                        help='List of defense to apply like tvm, quilting')
    parser.add_argument('--pretrained', default=False, action='store_true',
                        help='use pretrained model from model-zoo')

    # Defense params
    # TVM
    parser.add_argument('--tvm_weight', default=constants.TVM_WEIGHT,
                        type=float, metavar='N',
                        help='weight for TVM')
    parser.add_argument('--pixel_drop_rate', default=constants.PIXEL_DROP_RATE,
                        type=float, metavar='N',
                        help='Pixel drop rate to use in TVM')
    parser.add_argument('--tvm_method', default=constants.TVM_METHOD,
                        type=str, metavar='N',
                        help='Reconstruction method to use in TVM')
    # Quilting
    parser.add_argument('--quilting_patch_size',
                        default=constants.QUILTING_PATCH_SIZE,
                        type=int, metavar='N', help='Patch size to use in quilting')
    parser.add_argument('--quilting_neighbors', default=1, type=int, metavar='N',
                        help='Number of nearest neighbors to use for quilting patches')
    parser.add_argument('--quilting_random_stitch', default=False, action='store_true',
                        help='Randomly use quilting patches')
    # Quantization
    parser.add_argument('--quantize_depth', default=8, type=int, metavar='N',
                        help='Bit depth for quantization defense')

    return parser


def _setup_adversary_args(parser):
    # commaon params for generating or reading adversary images
    parser.add_argument('--n_samples', default=50000, type=int, metavar='N',
                        help='Max number of samples to test on')
    parser.add_argument('--attack_type', default=None, type=str, metavar='N',
                        help='Attack type (None(No attack) | blackbox | whitebox)')
    # parser.add_argument('--renormalize', default=False, action='store_true',
    #                     help='Renormalize for inception data params')
    parser.add_argument('--adversary', default=None, type=str, metavar='N',
                        help='Adversary to use for pre-generated attack images'
                        '(default = None)')
    parser.add_argument('--adversary_model', default='resnet50',
                        type=str, metavar='N',
                        help='Adversarial model to use (default resnet50)')
    parser.add_argument('--learning_rate', default=None, nargs='*',
                        type=float, metavar='N',
                        help='List of adversarial learning rate for each defense')
    parser.add_argument('--adv_strength', default=None, nargs='*',
                        type=float, metavar='N',
                        help='List of adversarial strength for each defense')
    parser.add_argument('--adversarial_root', default=DATA_ROOT + '/adversarial',
                        type=str, metavar='N',
                        help='Directory path adversary data')

    # params for generating adversary images
    parser.add_argument('--operation', default='transformation_on_adv',
                        type=str, metavar='N',
                        help='Operation to run (generate_adversarial, '
                        'concat_adversarial, compute_adversarial_stats)')
    parser.add_argument('--adversary_to_generate', default=None, type=str, metavar='N',
                        help='Adversary to generate (default = None)')
    parser.add_argument('--partition', default=0, type=int, metavar='N',
                        help='the data partition to work on (indexing from 0)')
    parser.add_argument('--partition_size', default=50000, type=int, metavar='N',
                        help='the size of each data partition')
    parser.add_argument('--data_type', default='train',
                        type=str, metavar='N',
                        help='data_type (train|raw) for transformation_on_raw')
    parser.add_argument('--max_adv_iter', default=10, type=int, metavar='N',
                        help='max iterations for iteratibe attacks')
    parser.add_argument('--fgs_mode', default=None,
                        type=str, metavar='N',
                        help='fgs_mode (logit | carlini) for loss computation in FGS')
    parser.add_argument('--margin', default=0, type=float, metavar='N',
                        help='margin parameter for cwl2')
    parser.add_argument('--compute_stats', default=False, action='store_true',
                        help='Compute adversarial stats(robustness, SSIM, '
                        'success rate)')
    parser.add_argument('--crop_frac', default=[1.0], nargs='*',
                        type=float, metavar='N',
                        help='crop fraction for ensembling or Carlini-Wagner')

    return parser


def _parse_train_opts():
    parser = argparse.ArgumentParser(description='Train convolutional network')
    parser = _setup_common_args(parser)

    parser.add_argument('--resume', default=False, action='store_true',
                        help='Resume training from checkpoint (if available)')
    parser.add_argument('--lr', default=None, type=float, metavar='N',
                        help='Initial learning rate for training, \
                        for inception_v4 use lr=0.045, 0.1 for others)')
    parser.add_argument('--lr_decay', default=None, type=float, metavar='N',
                        help='exponential learning rate decay(0.94 for \
                        inception_v4, 0.1 for others)')
    parser.add_argument('--lr_decay_stepsize', default=None, type=float, metavar='N',
                        help='decay learning rate after every stepsize \
                        epochs(2 for inception_v4, 30 for others)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='N',
                        help='amount of momentum (default = 0.9)')
    parser.add_argument('--weight_decay', default=1e-4, type=float, metavar='N',
                        help='amount of weight decay (default = 1e-4)')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='index of first epoch (default = 0)')
    parser.add_argument('--end_epoch', default=None, type=int, metavar='N',
                        help='index of last epoch (default = 90), \
                        for inception_v4 use end_epoch=160)')
    parser.add_argument('--preprocessed_epoch_data', default=False, action='store_true',
                        help='Randomly cropped data for each epoch is pre-generated')

    args = parser.parse_args()

    if args.model.startswith('inception'):
        hyperparams = constants.INCEPTION_V4_TRAINING_PARAMS
        # inception specific hyper_params
        args.rms_eps = constants.INCEPTION_V4_TRAINING_PARAMS['RMS_EPS']
        args.rms_alpha = constants.INCEPTION_V4_TRAINING_PARAMS['RMS_ALPHA']
    else:
        hyperparams = constants.TRAINING_PARAMS

    # init common training hyperparams
    if not args.lr:
        args.lr = hyperparams['LR']
    if not args.lr_decay:
        args.lr_decay = hyperparams['LR_DECAY']
    if not args.lr_decay_stepsize:
        args.lr_decay_stepsize = hyperparams['LR_DECAY_STEPSIZE']
    if not args.end_epoch:
        args.end_epoch = hyperparams['EPOCHS']

    return args


def _parse_classify_opts():
    # set input arguments:
    parser = argparse.ArgumentParser(description='Classify adversarial images')
    parser = _setup_common_args(parser)
    parser = _setup_adversary_args(parser)

    parser.add_argument('--ensemble', default=None, type=str, metavar='N',
                        help='ensembling type (None | avg | max)')
    parser.add_argument('--ncrops', default=None, nargs='*', type=int, metavar='N',
                        help='list of number of crops for each defense'
                        ' to use for ensembling')
    parser.add_argument('--crop_type', default=None, nargs='*',
                        type=str, metavar='N',
                        help='Crop during(center=CenterCrop, '
                        'random=RandomRisedCrop, sliding=Sliding Window Crops)')

    args = parser.parse_args()

    return args


# args for generating transformation data
def _parse_generate_opts():
    parser = argparse.ArgumentParser(description='Generate and save' +
                                                    'image transformations')
    parser = _setup_common_args(parser)
    parser = _setup_adversary_args(parser)

    # paths
    parser.add_argument('--out_dir', default=DATA_ROOT, type=str, metavar='N',
                        help='Directory path to output concatenated '
                        'transformed data')
    parser.add_argument('--partition_dir', default=DATA_ROOT + '/partitioned',
                        type=str, metavar='N',
                        help='Directory path to output transformed data')

    parser.add_argument('--data_batches', default=None, type=int, metavar='N',
                        help='Number of data batches to generate')

    parser.add_argument('--n_threads', default=20, type=int, metavar='N',
                        help='Number of threads for raw image transformation')

    parser.add_argument('--data_file', default=None, type=str, metavar='N',
                        help='Data file path to read images to visualize')

    args = parser.parse_args()

    return args


def _setup_model_based_data_params(args):
    if args.model.startswith('inception'):
        args.data_params = constants.INCEPTION_V4_DATA_PARAMS
    else:
        args.data_params = constants.RESNET_DENSENET_DATA_PARAMS

    return args


def _parse_adversarial_opts():
    parser = argparse.ArgumentParser(description='Generate adversarial images')
    parser = _setup_common_args(parser)
    parser = _setup_adversary_args(parser)
    args = parser.parse_args()
    return args


def _parse_quilting_patch_opts():
    # set input arguments:
    parser = argparse.ArgumentParser(description='Build FAISS index of patches')
    parser.add_argument('--patch_size', default=5, type=int, metavar='N',
                        help='size of patches (default: 5)')
    parser.add_argument('--num_patches', default=1000000, type=int, metavar='N',
                        help='number of patches in index (default: 1M)')
    parser.add_argument('--pca_dims', default=64, type=int, metavar='N',
                        help='number of pca dimensions to use (default: 64)')
    parser.add_argument('--patches_file', default='/tmp/tmp.pickle', type=str,
                        metavar='N', help='filename in which to save patches')
    parser.add_argument('--index_file', default='/tmp/tmp.faiss', type=str, metavar='N',
                        help='filename in which to save faiss index')
    args = parser.parse_args()
    return args


def parse_args(opt_type):
    assert isinstance(opt_type, OptType), \
        '{} not an instance of OptType Enum'.format(opt_type)
    assert DATA_ROOT, \
        "{} DATA_ROOT can't be empty. Update in path_config.py with correct value"

    if opt_type == OptType.QUILTING_PATCHES:
        args = _parse_quilting_patch_opts()
    else:
        if opt_type == OptType.TRAIN:
            args = _parse_train_opts()
        elif opt_type == OptType.CLASSIFY:
            args = _parse_classify_opts()
        elif opt_type == OptType.TRANSFORMATION:
            args = _parse_generate_opts()
        elif opt_type == OptType.ADVERSARIAL:
            args = _parse_adversarial_opts()

        # model
        assert args.model in constants.MODELS, \
            "model \"{}\" is not defined".format(args.model)

        args = _setup_model_based_data_params(args)

    # imagenet dir
    imagenet_dirs = [
        IMAGENET_DIR1,
        IMAGENET_DIR2,
    ]
    for path in imagenet_dirs:
        if os.path.isdir(path):
            args.imagenet_dir = str(path)
            break
        else:
            print("Can't find imagenet data at path: " + path)

    print("| Input args are:")
    print(args)

    return args
