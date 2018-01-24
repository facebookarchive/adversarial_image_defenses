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

from enum import Enum


class DefenseType(Enum):
    RAW = "raw"
    TVM = 'tvm'
    QUILTING = 'quilting'
    ENSEMBLE_TRAINING = 'ensemble_training'
    JPEG = 'jpeg'
    QUANTIZATION = 'quantize'

    @classmethod
    def has_value(cls, value):
        return (any(value == item.value for item in cls))

    def __str__(self):
        return str(self.value)


class AdversaryType(Enum):
    FGS = "fgs"
    IFGS = 'ifgs'
    CWL2 = 'cwl2'
    CWLINF = 'cwlinf'
    DEEPFOOL = 'deepfool'

    @classmethod
    def has_value(cls, value):
        return (any(value == item.value for item in cls))

    def __str__(self):
        return str(self.value)


class AttackType(Enum):
    WHITEBOX = "whitebox"
    BLACKBOX = 'blackbox'

    @classmethod
    def has_value(cls, value):
        return (any(value == item.value for item in cls))

    def __str__(self):
        return str(self.value)


# Constants
# Transformations params
QUILTING_PATCH_SIZE = 5
TVM_WEIGHT = 0.03
PIXEL_DROP_RATE = 0.5
# TODO: Get rid of all other methods
TVM_METHOD = 'bregman'

# Data params
INCEPTION_V4_DATA_PARAMS = {
    'MEAN_STD': {
        'MEAN': [0.5, 0.5, 0.5],
        'STD': [0.5, 0.5, 0.5],
    },
    'IMAGE_SIZE': 299,
    'IMAGE_SCALE_SIZE': 342,
    'NUM_CLASSES': 1000
}

RESNET_DENSENET_DATA_PARAMS = {
    'MEAN_STD': {
        'MEAN': [0.485, 0.456, 0.406],
        'STD': [0.229, 0.224, 0.225],
    },
    'IMAGE_SIZE': 224,
    'IMAGE_SCALE_SIZE': 256,
    'NUM_CLASSES': 1000
}

# Same as paper:https://arxiv.org/pdf/1602.07261.pdf
INCEPTION_V4_TRAINING_PARAMS = {
    'LR': 0.045,
    'LR_DECAY': 0.94,
    'LR_DECAY_STEPSIZE': 2,
    'EPOCHS': 160,
    'RMS_EPS': 1.0,
    'RMS_ALPHA': 0.9,
}

TRAINING_PARAMS = {
    'LR': 0.1,
    'LR_DECAY': 0.1,
    'LR_DECAY_STEPSIZE': 30,
    'EPOCHS': 90,
}

# List of supported models
MODELS = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
          'DenseNet', 'densenet121', 'densenet169', 'densenet201', 'densenet161',
          'Inception3', 'inception_v3', 'inception_v4']


# Read pre-calculated params for adversary for different settings
def get_adv_params(args, defense_idx=0):

    if args.attack_type is None:
        assert args.adv_strength is None and args.learning_rate is None

    if args.adversary_to_generate:
        adversary = args.adversary_to_generate
    else:
        adversary = args.adversary
    learning_rate = None
    adv_strength = None

    # get adv_strength from input arguments
    if args.adv_strength is not None:
        # No defense
        if args.defenses is None:
            assert (len(args.adv_strength) == 1 and
                    defense_idx == 0)
        else:
            # adv_strength is provided for each defense
            assert (len(args.defenses) == len(args.adv_strength) and
                    defense_idx < len(args.adv_strength))
        adv_strength = args.adv_strength[defense_idx]

    # get learning_rate from input arguments
    if args.learning_rate is not None:
        # No defense
        if args.defenses is None:
            assert (len(args.learning_rate) == 1 and
                    defense_idx == 0)
        else:
            # learning_rate is provided for each defense
            assert (len(args.defenses) == len(args.learning_rate) and
                    defense_idx < len(args.learning_rate))
        learning_rate = args.learning_rate[defense_idx]

    # if adversary params are not provided in input arguments,
    # then use below precomputed params on resnet50
    # parameters maintain L2 dissimilarity of ~0.06
    if adv_strength is None and learning_rate is None:
        assert (args.attack_type is None or
                AttackType.has_value(args.attack_type))
        consts = {adversary: (None, None)}
        # params for blackbox attack
        if args.attack_type == str(AttackType.BLACKBOX):
            consts = {
                str(AdversaryType.IFGS): (0.021, None),
                str(AdversaryType.DEEPFOOL): (0.96, None),
                str(AdversaryType.CWL2): (None, 31.5),
                str(AdversaryType.FGS): (None, 0.07),
            }
        elif args.attack_type == str(AttackType.WHITEBOX):
            if args.defenses[defense_idx] == str(DefenseType.TVM):
                consts = {
                    str(AdversaryType.IFGS): (0.018, None),
                    str(AdversaryType.DEEPFOOL): (3.36, None),
                    str(AdversaryType.CWL2): (None, 126),
                    str(AdversaryType.FGS): (None, 0.07),
                }

            elif args.defenses[defense_idx] == str(DefenseType.QUILTING):
                consts = {
                    str(AdversaryType.IFGS): (0.015, None),
                    str(AdversaryType.DEEPFOOL): (0.42, None),
                    str(AdversaryType.CWL2): (None, 17.4),
                    str(AdversaryType.FGS): (None, 0.07),
                }

            # If model used is InceptionResnetV2 using ensemble training
            elif args.defenses[defense_idx] == DefenseType.ENSEMBLE_TRAINING:
                consts = {
                    str(AdversaryType.IFGS): (0.01, None),
                    str(AdversaryType.DEEPFOOL): (1.1, None),
                    str(AdversaryType.CWL2): (None, 16.5),
                    str(AdversaryType.FGS): (None, 0.07),
                }
        return dict(zip(('learning_rate', 'adv_strength'), consts[adversary]))
    else:
        return {'learning_rate': learning_rate, 'adv_strength': adv_strength}
