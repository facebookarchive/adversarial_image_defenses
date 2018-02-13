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

try:
    import cPickle as pickle
except ImportError:
    import pickle
import progressbar
import random

import torch
import faiss

from lib.dataset import load_dataset, get_data_loader
import lib.opts as opts


# function that indexes a large number of patches:
def gather_patches(image_dataset, num_patches, patch_size, patch_transform=None):

    # assertions:
    assert isinstance(image_dataset, torch.utils.data.dataset.Dataset)
    assert type(num_patches) == int and num_patches > 0
    assert type(patch_size) == int and patch_size > 0
    if patch_transform is not None:
        assert callable(patch_transform)

    # gather patches (TODO: speed this up):
    patches, n = [], 0
    num_images = len(image_dataset)
    bar = progressbar.ProgressBar(num_patches)
    bar.start()
    data_loader = get_data_loader(image_dataset, batchsize=1, workers=1)
    for (img, _) in data_loader:
        img = img.squeeze()
        for _ in range(0, max(1, int(num_patches / num_images))):
            n += 1
            y = random.randint(0, img.size(1) - patch_size)
            x = random.randint(0, img.size(2) - patch_size)
            patch = img[:, y:y + patch_size, x:x + patch_size]
            if patch_transform is not None:
                patch = patch_transform(patch)
            patches.append(patch)
            if n % 100 == 0:
                bar.update(n)
            if n >= num_patches:
                break
        if n >= num_patches:
            break

    # copy all patches into single tensor:
    patches = torch.stack(patches, dim=0)
    patches = patches.view(patches.size(0), int(patches.nelement() / patches.size(0)))
    return patches


# function that trains faiss index on patches and saves them:
def index_patches(patches, index_file, pca_dims=64):

    # settings for faiss:
    num_lists, M, num_bits = 200, 16, 8

    # assertions:
    assert torch.is_tensor(patches) and patches.dim() == 2
    assert type(pca_dims) == int and pca_dims > 0
    if pca_dims > patches.size(1):
        print('WARNING: Input dimension < %d. Using fewer PCA dimensions.' % pca_dims)
        pca_dims = patches.size(1) - (patches.size(1) % M)

    # construct faiss index:
    quantizer = faiss.IndexFlatL2(pca_dims)
    assert pca_dims % M == 0
    sub_index = faiss.IndexIVFPQ(quantizer, pca_dims, num_lists, M, num_bits)
    pca_matrix = faiss.PCAMatrix(patches.size(1), pca_dims, 0, True)
    faiss_index = faiss.IndexPreTransform(pca_matrix, sub_index)

    # train faiss index:
    patches = patches.numpy()
    faiss_index.train(patches)
    faiss_index.add(patches)

    # save faiss index:
    print('| writing faiss index to %s' % index_file)
    faiss.write_index(faiss_index, index_file)


# run all the things:
def create_faiss_patches(args):

    # load image dataset:
    print('| set up image loader...')
    image_dataset = load_dataset(args, 'train', None, with_transformation=True)
    image_dataset.imgs = image_dataset.imgs[:20000]  # we don't need all images

    # gather image patches:
    print('| gather image patches...')
    patches = gather_patches(
        image_dataset, args.num_patches, args.quilting_patch_size,
        patch_transform=None,
    )

    # build faiss index:
    print('| training faiss index...')
    index_patches(patches, args.index_file, pca_dims=args.pca_dims)

    # save patches:
    with open(args.patches_file, 'wb') as fwrite:
        print('| writing patches to %s' % args.patches_file)
        pickle.dump(patches, fwrite, pickle.HIGHEST_PROTOCOL)


# run:
if __name__ == '__main__':
    # parse input arguments:
    args = opts.parse_args(opts.OptType.QUILTING_PATCHES)
    create_faiss_patches(args)
