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

import pkgutil
if pkgutil.find_loader("adversarial") is not None:
    # If module is installed using "pip install ."
    from adversarial.gen_adversarial_images import generate_adversarial_images
    from adversarial.gen_transformed_images import generate_transformed_images
    from adversarial.classify_images import classify_images
    from adversarial.train_model import train_model
else:
    from gen_adversarial_images import generate_adversarial_images
    from gen_transformed_images import generate_transformed_images
    from classify_images import classify_images
    from train_model import train_model

from lib import opts
import os


# Generate adversarial images
def _generate_adversarial_images():
    # load default args for adversary functions
    args = opts.parse_args(opts.OptType.ADVERSARIAL)
    # edit default args
    args.operation = "generate_adversarial"
    args.model = "resnet50"
    args.adversary_to_generate = "fgs"
    args.defenses = None
    args.partition_size = 1  # Number of samples to generate
    args.n_samples = 10000  # Total samples in input data
    args.data_type = "val"  # input dataset type
    args.normalize = True  # apply normalization on input data
    args.attack_type = "blackbox"  # For <whitebox> attack, use transformed models
    args.pretrained = True  # Use pretrained model from model-zoo

    generate_adversarial_images(args)


# Apply transformations
def _generate_transformed_images():
    # load default args for transformation functions
    args = opts.parse_args(opts.OptType.TRANSFORMATION)
    # edit default args
    # Apply transformations on raw images,
    #  for adversarial images use "transformation_on_adv"
    args.operation = "transformation_on_raw"
    args.adversary = None  # update to adversary for operation "transformation_on_adv"
    # For quilting expects patches data at QUILTING_ROOT (defined in
    #  path_config.json or passed in args)
    args.defenses = ["quilting"]  # <"tvm", "quilting", "jpeg", quantize>
    args.partition_size = 1  # Number of samples to generate
    args.data_type = "val"  # input dataset type

    # args.n_samples = 50000  # Total samples in input data when reading from .pth files
    # args.attack_type = "blackbox"  # Used for file paths for "transformation_on_adv"

    generate_transformed_images(args)
    print('Transformed images saved at {}'.format(
        os.path.join(args.partition_dir, args.defenses[0])))


def _classify_images():
    # classify images without any attack or defense
    args = opts.parse_args(opts.OptType.CLASSIFY)

    # edit default args
    args.n_samples = 1  # Total samples in input data
    args.normalize = True  # apply normalization on input data
    args.pretrained = True  # Use pretrained model from model-zoo
    # To classify transformed images using transformed model update defenses to
    #  <tvm|quilting>
    args.defenses = None
    # To classify attack images update attack_type to <blackbox|whitebox>
    args.attack_type = None
    # To classify attack images update adversary to <fgs|ifgs|cwl2\deepfool>
    args.adversary = None

    classify_images(args)


def _train_model():
    args = opts.parse_args(opts.OptType.TRAIN)

    # edit default args
    # To classify transformed images using transformed model update defenses to
    #  <tvm|quilting>
    args.defenses = None  # defense=<(raw, tvm, quilting, jpeg, quantization)>
    args.model = "resnet50"
    args.normalize = True  # apply normalization on input data
    args.resume = True  # Resume training from checkpoint if available

    train_model(args)


if __name__ == '__main__':
    _generate_adversarial_images()
    _generate_transformed_images()
    _train_model()
    _classify_images()
