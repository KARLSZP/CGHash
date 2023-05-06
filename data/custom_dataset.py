"""
    Part of Codes are forked from other work(s).
    Links and Reference would be added in open-source version.
"""
import numpy as np
import torch
from torch.utils.data import Dataset


class AugmentedDataset(Dataset):
    """
        AugmentedDataset
        Returns an image with one of its strong augmentation.
    """

    def __init__(self, dataset):
        super(AugmentedDataset, self).__init__()
        transform = dataset.transform
        dataset.transform = None
        self.dataset = dataset

        if isinstance(transform, dict):
            self.image_transform = transform['standard']
            self.augmentation_transform = transform['augment']

        else:
            self.image_transform = transform
            self.augmentation_transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset.__getitem__(index)
        image = sample['image']

        sample['image'] = self.image_transform(image)
        sample['image_augmented'] = self.augmentation_transform(image)

        return sample


class NeighborsDataset(Dataset):
    """
        NeighborsDataset
        Returns an image with one of its neighbors.
    """

    def __init__(self, dataset, indices, num_neighbors=None):
        super(NeighborsDataset, self).__init__()
        transform = dataset.transform

        if isinstance(transform, dict):
            self.anchor_transform = transform['standard']
            self.neighbor_transform = transform['augment']
        else:
            self.anchor_transform = transform
            self.neighbor_transform = transform

        dataset.transform = None
        self.dataset = dataset
        # Nearest neighbor indices (np.array  [len(dataset) x k])
        self.indices = indices
        if num_neighbors is not None:
            self.indices = self.indices[:, :num_neighbors+1]
        assert (self.indices.shape[0] == len(self.dataset))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        output = {}
        anchor = self.dataset.__getitem__(index)
        output['anchor'] = self.anchor_transform(anchor['image'])
        output['augment'] = self.neighbor_transform(anchor['image'])

        neighbor_index = np.random.choice(self.indices[index], 1)[0]
        neighbor = self.dataset.__getitem__(neighbor_index)
        neighbor['image'] = self.neighbor_transform(neighbor['image'])
        output['neighbor'] = neighbor['image']

        output['possible_neighbors'] = torch.from_numpy(self.indices[index])
        output['target'] = anchor['target']
        if 'multi_target' in anchor.keys():
            output['multi_target'] = anchor['multi_target']
        output['index'] = index, neighbor_index
        return output
