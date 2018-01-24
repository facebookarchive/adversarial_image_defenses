
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from torchvision.datasets.folder import ImageFolder, default_loader


class PartialImageFolder(ImageFolder):
    def __init__(self, root, data_indices=None,
                    transform=None, target_transform=None, loader=default_loader):
        super(PartialImageFolder, self).__init__(
            root, transform=transform, target_transform=target_transform, loader=loader)

        if data_indices:
            assert ('start_idx' in data_indices and
                    isinstance(data_indices['start_idx'], int)), \
                "data_indices expects argument start_idx of int type"
            assert ('end_idx' in data_indices and
                    isinstance(data_indices['end_idx'], int)), \
                "data_indices expects argument end_idx of int type"
            assert data_indices['start_idx'] < len(self.imgs)

            end_idx = min(data_indices['end_idx'], len(self.imgs))
            self.imgs = self.imgs[data_indices['start_idx']:end_idx]

    def get_classes(self):
        return self.classes
