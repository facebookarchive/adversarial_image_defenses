#!/usr/bin/env python2

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import ctypes
import torch

# load seam-finding library:
FINDSEAM_LIB = ctypes.cdll.LoadLibrary(
    'libexperimental_deeplearning_lvdmaaten_adversarial_findseam.so')

# other globals:
LATTICE_CACHE = {}  # cache lattices here


# function that constructs a four-connected lattice:
def __four_lattice__(height, width, use_cache=True):

    # try the cache first:
    if use_cache and (height, width) in LATTICE_CACHE:
        return LATTICE_CACHE[(height, width)]

    # assertions and initialization:
    assert type(width) == int and type(height) == int and \
        width > 0 and height > 0, 'height and width should be positive integers'
    N = height * width
    height, width = width, height  # tensors are in row-major format
    graph = {
        'from': torch.LongTensor(4 * N - (height + width) * 2),
        'to': torch.LongTensor(4 * N - (height + width) * 2),
    }

    # closure that copies stuff in:
    def add_edges(i, j, offset):
        graph['from'].narrow(0, offset, i.nelement()).copy_(i)
        graph['from'].narrow(0, offset + i.nelement(), j.nelement()).copy_(j)
        graph['to'].narrow(0, offset, j.nelement()).copy_(j)
        graph['to'].narrow(0, offset + j.nelement(), i.nelement()).copy_(i)

    # add vertical connections:
    i = torch.arange(0, N).squeeze().long()
    mask = torch.ByteTensor(N).fill_(1)
    mask.index_fill_(0, torch.arange(height - 1, N, height).squeeze().long(), 0)
    i = i[mask]
    add_edges(i, torch.add(i, 1), 0)

    # add horizontal connections:
    offset = 2 * i.nelement()
    i = torch.arange(0, N - height).squeeze().long()
    add_edges(i, torch.add(i, height), offset)

    # cache and return graph:
    if use_cache:
        LATTICE_CACHE[(height, width)] = graph
    return graph


# utility function for checking inputs:
def __assert_inputs__(im1, im2, mask=None):
    assert type(im1) == torch.ByteTensor or type(im1) == torch.FloatTensor, \
        'im1 should be a ByteTensor or FloatTensor'
    assert type(im2) == torch.ByteTensor or type(im2) == torch.FloatTensor, \
        'im2 should be a ByteTensor or FloatTensor'
    assert im1.dim() == 3, 'im1 should be three-dimensional'
    assert im2.dim() == 3, 'im2 should be three-dimensional'
    assert im1.size() == im2.size(), 'im1 and im2 should have same size'
    if mask is not None:
            assert mask.dim() == 2, 'mask should be two-dimensional'
            assert type(mask) == torch.ByteTensor, 'mask should be torch.ByteTensor'
            assert mask.size(0) == im1.size(1) and mask.size(1) == im1.size(2), \
                'mask should have same height and width as images'


# function that finds seam between two images:
def find_seam(im1, im2, mask):

    # assertions:
    __assert_inputs__(im1, im2, mask)
    im1 = im1.float()
    im2 = im2.float()

    # construct edge weights:
    graph = __four_lattice__(im1.size(1), im1.size(2))
    values = torch.FloatTensor(graph['from'].size(0)).fill_(0.)
    for c in range(im1.size(0)):
        im1c = im1[c].contiguous().view(im1.size(1) * im1.size(2))
        im2c = im2[c].contiguous().view(im2.size(1) * im2.size(2))
        values.add_(torch.abs(
            im2c.index_select(0, graph['to']) -
            im1c.index_select(0, graph['from'])
        ))

    # construct terminal weights:
    idxim = torch.arange(0, mask.nelement()).long().view(mask.size())
    tvalues = torch.FloatTensor(mask.nelement(), 2).fill_(0)
    for c in range(2):
        select_c = (mask == (c + 1))
        if select_c.any():
            tvalues.select(1, c).index_fill_(0, idxim[select_c], float('inf'))

    # convert graph to IntTensor (make sure this is not GC'ed):
    graph_from = graph['from'].int()
    graph_to = graph['to'].int()

    # run the Boykov algorithm to obtain stitching mask:
    labels = torch.IntTensor(mask.nelement())
    FINDSEAM_LIB.findseam(
        ctypes.c_int(mask.nelement()),
        ctypes.c_int(values.nelement()),
        ctypes.c_void_p(graph_from.data_ptr()),
        ctypes.c_void_p(graph_to.data_ptr()),
        ctypes.c_void_p(values.data_ptr()),
        ctypes.c_void_p(tvalues.data_ptr()),
        ctypes.c_void_p(labels.data_ptr()),
    )
    mask = labels.resize_(mask.size()).byte()
    return mask


# function that performs the stitch:
def __stitch__(im1, im2, overlap, y, x):

    # assertions:
    __assert_inputs__(im1, im2)

    # construct mask:
    patch_size = im1.size(1)
    mask = torch.ByteTensor(patch_size, patch_size).fill_(2)
    if y > 0:  # there is not overlap at the border
        mask.narrow(0, 0, overlap).fill_(0)
    if x > 0:  # there is not overlap at the border
        mask.narrow(1, 0, overlap).fill_(0)

    # seam the two patches:
    seam_mask = find_seam(im1, im2, mask)
    stitched_im = im1.clone()
    for c in range(stitched_im.size(0)):
        stitched_im[c][seam_mask == 1] = im2[c][seam_mask]
    return stitched_im


# main quilting function:
def quilting(img, faiss_index, patch_dict, patch_size=5, overlap=2,
                                        graphcut=False, patch_transform=None):

    # assertions:
    assert torch.is_tensor(img)
    assert torch.is_tensor(patch_dict) and patch_dict.dim() == 2
    assert type(patch_size) == int and patch_size > 0
    assert type(overlap) == int and overlap > 0
    assert patch_size > overlap
    if patch_transform is not None:
        assert callable(patch_transform)

    # gather all image patches:
    patches = []
    y_range = range(0, img.size(1) - patch_size, patch_size - overlap)
    x_range = range(0, img.size(2) - patch_size, patch_size - overlap)
    for y in y_range:
        for x in range(0, img.size(2) - patch_size, patch_size - overlap):
            patch = img[:, y:y + patch_size, x:x + patch_size]
            if patch_transform is not None:
                patch = patch_transform(patch)
            patches.append(patch)

    # find nearest patches in faiss index:
    patches = torch.stack(patches, dim=0)
    patches = patches.view(patches.size(0), int(patches.nelement() / patches.size(0)))
    faiss_index.nprobe = 5
    _, neighbors = faiss_index.search(patches.numpy(), 1)
    neighbors = torch.LongTensor(neighbors).squeeze()
    if (neighbors == -1).any():
        print('WARNING: %d out of %d neighbor searches failed.' %
              ((neighbors == -1).sum(), neighbors.nelement()))

    # piece the image back together:
    n = 0
    quilt_img = img.clone().fill_(0)
    for y in y_range:
        for x in x_range:
            if neighbors[n] != -1:

                # get current image and new patch:
                patch = patch_dict[neighbors[n]].view(
                    img.size(0), patch_size, patch_size
                )
                cur_img = quilt_img[:, y:y + patch_size, x:x + patch_size]

                # compute graph cut if requested:
                if graphcut:
                    patch = __stitch__(cur_img, patch, overlap, y, x)

                # copy the patch into the image:
                cur_img.copy_(patch)
            n += 1

    # return the quilted image:
    return quilt_img
