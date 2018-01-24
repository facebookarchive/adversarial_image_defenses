#!/usr/bin/env python2

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.utils.data


class TransformDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, transform=None, data_indices=None):
        super(TransformDataset, self).__init__()
        # assertions
        assert isinstance(dataset, torch.utils.data.Dataset)
        if transform is not None:
            assert callable(transform)
        if data_indices:
            assert ('start_idx' in data_indices and
                    isinstance(data_indices['start_idx'], int)), \
                "data_indices expects argument start_idx of int type"
            assert ('end_idx' in data_indices and
                    isinstance(data_indices['end_idx'], int)), \
                "data_indices expects argument end_idx of int type"
            assert data_indices['start_idx'] < len(dataset)

        self.dataset = dataset
        self.transform = transform
        if data_indices:
            end_idx = min(data_indices['end_idx'], len(dataset))
            self.dataset.data_tensor = self.dataset.data_tensor[data_indices['start_idx']:end_idx]
            self.dataset.target_tensor = self.dataset.target_tensor[data_indices['start_idx']:end_idx]

    # Apply each transform on an image and concatenate output into multichannel
    def __getitem__(self, idx):
        item = self.dataset[idx]  # sample is a tuple of form (img, target)
        if self.transform is not None:
            item = (self.transform(item[0]), item[1])
        return item

    def __len__(self):
        return len(self.dataset)
