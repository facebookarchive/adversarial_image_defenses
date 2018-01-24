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

import cPickle as pickle
import math

import torch
from lib.transformations.quilting_fast import quilting
from lib.transformations.tvm import reconstruct as tvm
from PIL import Image
from lib.paths import get_quilting_filepaths
from torchvision.transforms import ToPILImage, ToTensor
try:
    from cStringIO import StringIO as BytesIO
except ImportError:
    from io import BytesIO
from lib.constants import DefenseType


def _quantize_img(im, depth=8):
    assert torch.is_tensor(im)
    N = int(math.pow(2, depth))
    im = (im * N).round()
    im = im / N
    return im


def _jpeg_compression(im):
    assert torch.is_tensor(im)
    im = ToPILImage()(im)
    savepath = BytesIO()
    im.save(savepath, 'JPEG', quality=75)
    im = Image.open(savepath)
    im = ToTensor()(im)
    return im


# class describing a defense transformation:
class Defense(object):

    def __init__(self, defense, defense_name):
        assert callable(defense)
        self.defense = defense
        self.defense_name = defense_name

    def __call__(self, im):
        return self.defense(im)

    def get_name(self):
        return self.defense_name


# function that returns defense:
def get_defense(defense_name, args):

    print('| Defense: {}'.format(defense_name))
    assert (defense_name is None or DefenseType.has_value(defense_name)), (
        "{} defense type not defined".format(defense_name))

    defense = None
    # set up quilting defense:
    if defense_name == str(DefenseType.RAW):
        # return image as it is
        def defense(im):
            return im
        defense = Defense(defense, defense_name)

    elif defense_name == str(DefenseType.QUILTING):
        # load faiss index:
        import faiss
        print('| load quilting patch data...')
        patches_filename, index_filename = get_quilting_filepaths(args)
        with open(patches_filename, 'rb') as fread:
            patches = pickle.load(fread)
            patch_size = int(math.sqrt(patches.size(1) / 3))
        faiss_index = faiss.read_index(index_filename)

        # the actual quilting defense:
        def defense(im):
            im = quilting(
                im, faiss_index, patches,
                patch_size=patch_size,
                overlap=(patch_size // 2),
                graphcut=True,
                k=args.quilting_neighbors,
                random_stitch=args.quilting_random_stitch
            )
            # Clamping because some values are overflowing in quilting
            im = torch.clamp(im, min=0.0, max=1.0)
            return im
        defense = Defense(defense, defense_name)

    # set up tvm defense:
    elif defense_name == str(DefenseType.TVM):
        # the actual tvm defense:
        def defense(im):
            im = tvm(
                im,
                args.pixel_drop_rate,
                args.tvm_method,
                args.tvm_weight
            )
            return im
        defense = Defense(defense, defense_name)

    elif defense_name == str(DefenseType.QUANTIZATION):
        def defense(im):
            im = _quantize_img(im, depth=args.quantize_depth)
            return im
        defense = Defense(defense, defense_name)

    elif defense_name == str(DefenseType.JPEG):
        def defense(im):
            im = _jpeg_compression(im)
            return im
        defense = Defense(defense, defense_name)

    else:
        print('| No defense for \"%s\" is available' % (defense_name))

    return defense
