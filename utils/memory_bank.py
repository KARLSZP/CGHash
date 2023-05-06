"""
    Part of Codes are forked from other work(s).
    Links and Reference would be added in open-source version.
"""
import numpy as np
import torch

from tqdm import tqdm


class MemoryBank(object):
    def __init__(self, n, dim, num_classes, temperature):
        self.n = n
        self.dim = dim
        self.C = num_classes
        self.device = 'cpu'
        self.temperature = temperature
        self.ptr = 0

        self.features = torch.FloatTensor(self.n, self.dim)
        self.targets = torch.LongTensor(self.n)
        self.multi_targets = torch.LongTensor(self.n, num_classes)

    def mine_nearest_neighbors(self, topk, multilabel=False, calculate_accuracy=True):
        # mine the topk nearest neighbors for every sample
        scores = torch.mm(self.features, self.features.T).cpu()
        # BUG? appear in torch.topk: returning overflow values when using GPU
        _, indices = scores.topk(k=topk+1, dim=1)
        # indices = indices.cpu()

        # evaluate
        if calculate_accuracy:
            # Exclude sample itself for eval
            if multilabel:
                targets = self.multi_targets.cpu().numpy()
                anchor_targets = np.repeat(
                    targets.reshape(-1, 1, self.C), topk, axis=1)
                neighbor_targets = np.take(targets, indices[:, 1:], axis=0)
                accuracy = np.mean(
                    (neighbor_targets * anchor_targets).sum(axis=2) > 0)
            else:
                targets = self.targets.cpu().numpy()
                neighbor_targets = np.take(targets, indices[:, 1:], axis=0)
                anchor_targets = np.repeat(
                    targets.reshape(-1, 1), topk, axis=1)
                accuracy = np.mean(neighbor_targets == anchor_targets)
            return indices, accuracy

        else:
            return indices

    @torch.no_grad()
    def fill_memory_bank(self, loader, model, multilabel=False):
        model.eval()
        self.reset()

        for i, batch in enumerate(tqdm(loader)):
            images = batch['image'].cuda(non_blocking=True)
            if multilabel:
                targets = batch['multi_target'].cuda(non_blocking=True)
            else:
                targets = batch['target'].cuda(non_blocking=True)
            output = model(images)
            self.update(output, targets)

    def reset(self):
        self.ptr = 0

    def update(self, features, targets):
        b = features.size(0)

        assert (b + self.ptr <= self.n)

        self.features[self.ptr:self.ptr+b].copy_(features.detach())
        if targets.dim() > 1:
            self.multi_targets[self.ptr:self.ptr+b].copy_(targets.detach())
        else:
            self.targets[self.ptr:self.ptr+b].copy_(targets.detach())
        self.ptr += b

    def to(self, device):
        self.features = self.features.to(device)
        self.targets = self.targets.to(device)
        self.device = device
        return self

    def cpu(self):
        self.to('cpu')
        return self

    def cuda(self):
        self.to('cuda')
        return self
