import pickle as pkl

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from utils.data_path import DataPath


class COCO(Dataset):
    def __init__(self, root=DataPath.db_root_dir('coco'), split="train", transform=None):

        super(COCO, self).__init__()
        self.root = root
        self.transform = transform

        self.data, targets = self.load_coco(split=split)
        self.multi_targets = torch.from_numpy(targets)
        self.targets = self.multi_targets.argmax(dim=1)
        self.num_classes = self.multi_targets.shape[1]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            dict: {'image': image, 'target': index of target class, 'meta': dict}
        """
        img, target = self.data[index], self.targets[index]
        multi_target = self.multi_targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        out = {'image': img, 'target': target,
               'multi_target': multi_target}
        return out

    def get_image(self, index):
        img = self.data[index]
        return img

    def __len__(self):
        return len(self.data)

    def load_coco(self, split):
        coco_path = self.root
        if not coco_path.endswith("/"):
            coco_path = coco_path + "/"
        with open(coco_path + "X_{}.pk".format(split), "rb") as f:
            X = pkl.load(f)
        # with open(coco_path + "Y_{}.pk".format(split), "rb") as f:
        #     Y = pkl.load(f)
        # X = np.load(coco_path + "X_{}.npz".format(split), allow_pickle=True)
        Y = np.load(coco_path + "Y_{}.npz".format(split), allow_pickle=True)
        print("X_{}.pk loaded.".format(split))
        return X, Y
        # X_train = np.load(coco_path + "X_train.npz", allow_pickle=True)
        # Y_train = np.load(coco_path + "Y_train.npz", allow_pickle=True)
        # X_val = np.load(coco_path + "X_val.npz", allow_pickle=True)
        # Y_val = np.load(coco_path + "Y_val.npz", allow_pickle=True)
        # X_test = np.load(coco_path + "X_test.npz", allow_pickle=True)
        # Y_test = np.load(coco_path + "Y_test.npz", allow_pickle=True)
        # X_database = np.load(coco_path + "X_database.npz", allow_pickle=True)
        # Y_database = np.load(coco_path + "Y_database.npz", allow_pickle=True)

        # return X_train, Y_train, X_val, Y_val, X_test, Y_test, X_database, Y_database


if __name__ == "__main__":
    dataset = COCO(split="train", transform=None)
    print(len(dataset))
