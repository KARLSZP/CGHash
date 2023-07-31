import collections
import math
import os
import random

import numpy as np
import torch
import torchvision.transforms as transforms
from data.augment import Augment, Cutout
from termcolor import colored
from torch.utils.data import DataLoader


def get_criterion(p):
    if p['criterion'] == 'cghash':
        from losses.losses import ConsistentLoss, EntropyLoss, MaskedContLoss
        criterion = {
            'consist': ConsistentLoss(**p['criterion_kwargs']['consistent']),
            'MI': EntropyLoss(**p['criterion_kwargs']['entropy']),
            'mcont': MaskedContLoss(**p['criterion_kwargs']['mcont']),
        }
        criterion_state = {
            'consist': (p['criterion_kwargs']['consistent']['consistent_w'] != 0 or
                        p['criterion_kwargs']['consistent']['inconsistent_w'] != 0),
            'MI': (p['criterion_kwargs']['entropy']['entropy_w'] != 0 or
                   p['criterion_kwargs']['entropy']['sharp_w'] != 0),
            'mcont': (p['criterion_kwargs']['mcont']['w'] != 0),
        }

    elif p['criterion'] == 'selflabel':
        from losses.losses import ConfidenceBasedCE
        criterion = {
            'confidenceCE': ConfidenceBasedCE(**p['criterion_kwargs']['confidenceCE']),
        }
        criterion_state = {
            'confidenceCE': True
        }

    else:
        raise ValueError('Invalid criterion {}'.format(p['criterion']))

    return criterion, criterion_state


def get_feature_dimensions_backbone(p):
    if p['backbone'] == 'resnet18':
        return 512

    elif p['backbone'] == 'resnet50':
        return 2048

    else:
        raise NotImplementedError


def get_backbone(p):
    if p['backbone'] == 'resnet18':
        if p['train_db_name'] in 'cifar-10':
            from models.resnet_cifar import resnet18
            backbone = resnet18()
        else:
            raise NotImplementedError

    elif p['backbone'] == 'resnet50':
        if p['train_db_name'] == 'coco':
            from models.resnet import resnet50
            backbone = resnet50()
        elif p['train_db_name'] == 'nus':
            from models.resnet import resnet50
            backbone = resnet50()
        else:
            raise NotImplementedError

    else:
        raise ValueError('Invalid backbone {}'.format(p['backbone']))

    return backbone


def get_model(p, pretrain_path=None):
    # Get backbone
    backbone = get_backbone(p)

    # Setup
    if p['setup'] in ['simclr', 'moco']:
        from models.models import ContrastiveModel
        model = ContrastiveModel(backbone, **p['model_kwargs'])

    elif p['setup'] in ['cghash', 'selflabel']:
        from models.models import CGHashModel
        model = CGHashModel(
            backbone, p['num_classes'], p['encode_length'], p['tau'])

    else:
        raise ValueError('Invalid setup {}'.format(p['setup']))

    # Load pretrained weights
    if pretrain_path is not None and os.path.exists(pretrain_path):
        print("Loading from {}...".format(pretrain_path))
        state = torch.load(pretrain_path, map_location='cpu')

        if p['setup'] == 'simclr':
            missing = model.load_state_dict(state, strict=False)

        elif p['setup'] == 'moco':
            raise NotImplementedError

        elif p['setup'] == 'cghash':
            for k in list(state.keys()):
                state[k[7:]] = state.pop(k)
            missing = model.load_state_dict(state, strict=False)
            assert (set(missing[1]) == {
                'contrastive_head.0.weight', 'contrastive_head.0.bias',
                'contrastive_head.2.weight', 'contrastive_head.2.bias'}
                or set(missing[1]) == {
                'contrastive_head.weight', 'contrastive_head.bias'})

        elif p['setup'] == 'selflabel':
            model_state = state['model']
            missing = model.load_state_dict(model_state, strict=True)
            print(missing)

        else:
            raise NotImplementedError

    elif pretrain_path is not None and not os.path.exists(pretrain_path):
        raise ValueError(
            'Path with pre-trained weights does not exist {}'.format(pretrain_path))

    return model


def get_base_dataset(p, transform):
    # Base dataset
    if p['train_db_name'] == 'cifar-10':
        from data.cifar import CIFAR10
        dataset = CIFAR10(train=True, transform=transform)
    elif p['train_db_name'] == 'coco':
        from data.coco import COCO
        dataset = COCO(split="database", transform=transform)
    elif p['train_db_name'] == 'nus':
        from data.nus import NUS
        dataset = NUS(split="database", transform=transform)
    else:
        raise ValueError('Invalid train dataset {}'.format(p['train_db_name']))

    return dataset


def get_train_dataset(p, transform, to_augmented_dataset=False, to_neighbors_dataset=False):
    # Base dataset
    if p['train_db_name'] == 'cifar-10':
        from data.cifar import CIFAR10
        dataset = CIFAR10(train=True, transform=transform, download=True)

    elif p['train_db_name'] == 'coco':
        from data.coco import COCO
        dataset = COCO(split="train", transform=transform)

    elif p['train_db_name'] == 'nus':
        from data.nus import NUS
        dataset = NUS(split="train", transform=transform)

    else:
        raise ValueError('Invalid train dataset {}'.format(p['train_db_name']))

    if to_neighbors_dataset:
        from data.custom_dataset import NeighborsDataset
        indices = np.load(p['topk_neighbors_train_path'])
        dataset = NeighborsDataset(dataset, indices, p['num_neighbors'])

    # self-labeling
    if to_augmented_dataset:
        from data.custom_dataset import AugmentedDataset
        dataset = AugmentedDataset(dataset)

    return dataset


def get_val_dataset(p, transform=None):
    # Base dataset
    if p['val_db_name'] == 'cifar-10':
        from data.cifar import CIFAR10
        dataset = CIFAR10(train=False, transform=transform, download=True)

    elif p['train_db_name'] == 'coco':
        from data.coco import COCO
        dataset = COCO(split="val", transform=transform)

    elif p['train_db_name'] == 'nus':
        from data.nus import NUS
        dataset = NUS(split="val", transform=transform)

    else:
        raise ValueError(
            'Invalid validation dataset {}'.format(p['val_db_name']))

    return dataset


def get_base_dataloader(p, dataset):
    return DataLoader(dataset, num_workers=p['num_workers'], pin_memory=False,
                      batch_size=p['batch_size'], collate_fn=collate_custom,
                      drop_last=False, shuffle=False, persistent_workers=False)


def get_train_dataloader(p, dataset, sampler=None):
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = None
    return DataLoader(dataset, num_workers=p['num_workers'], pin_memory=True,
                      batch_size=p['batch_size'], collate_fn=collate_custom,
                      drop_last=True, shuffle=(sampler is None), sampler=sampler,
                      worker_init_fn=seed_worker, generator=g,
                      prefetch_factor=1, persistent_workers=True)


def get_val_dataloader(p, dataset):
    return DataLoader(dataset, num_workers=p['num_workers'],
                      batch_size=p['batch_size'], pin_memory=True, collate_fn=collate_custom,
                      drop_last=False, shuffle=False, persistent_workers=True)


def get_train_transformations(p):
    if p['augmentation_strategy'] == 'standard':
        # Standard augmentation strategy
        return transforms.Compose([
            transforms.RandomResizedCrop(
                **p['augmentation_kwargs']['random_resized_crop']),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(**p['augmentation_kwargs']['normalize'])
        ])

    elif p['augmentation_strategy'] == 'simclr':
        # Augmentation strategy from the SimCLR paper
        return transforms.Compose([
            transforms.RandomResizedCrop(
                **p['augmentation_kwargs']['random_resized_crop']),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(
                    **p['augmentation_kwargs']['color_jitter'])
            ], p=p['augmentation_kwargs']['color_jitter_random_apply']['p']),
            transforms.RandomGrayscale(
                **p['augmentation_kwargs']['random_grayscale']),
            transforms.ToTensor(),
            transforms.Normalize(**p['augmentation_kwargs']['normalize'])
        ])

    elif p['augmentation_strategy'] == 'cutout':
        # Augmentation strategy from the SCAN paper
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(p['augmentation_kwargs']['crop_size']),
            Augment(p['augmentation_kwargs']['num_strong_augs']),
            transforms.ToTensor(),
            transforms.Normalize(**p['augmentation_kwargs']['normalize']),
            Cutout(
                n_holes=p['augmentation_kwargs']['cutout_kwargs']['n_holes'],
                length=p['augmentation_kwargs']['cutout_kwargs']['length'],
                _random=p['augmentation_kwargs']['cutout_kwargs']['random'])])
    else:
        raise ValueError('Invalid augmentation strategy {}'.format(
            p['augmentation_strategy']))


def get_strong_train_transformation(p):
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(p['strong_augmentation_kwargs']['crop_size']),
        Augment(p['strong_augmentation_kwargs']['num_strong_augs']),
        transforms.ToTensor(),
        transforms.Normalize(**p['strong_augmentation_kwargs']['normalize']),
        Cutout(
            n_holes=p['strong_augmentation_kwargs']['cutout_kwargs']['n_holes'],
            length=p['strong_augmentation_kwargs']['cutout_kwargs']['length'],
            _random=p['strong_augmentation_kwargs']['cutout_kwargs']['random'])])


def get_val_transformations(p):
    if p['train_db_name'] == 'cifar-10':
        return transforms.Compose([
            transforms.Resize(p['transformation_kwargs']['crop_size']),
            transforms.ToTensor(),
            transforms.Normalize(**p['transformation_kwargs']['normalize'])])
    else:
        return transforms.Compose([
            transforms.CenterCrop(p['transformation_kwargs']['crop_size']),
            transforms.ToTensor(),
            transforms.Normalize(**p['transformation_kwargs']['normalize'])])


def get_optimizer(p, model, finetune_backbone=True):
    if not p['finetune_backbone']:
        print(colored(' | Not fine-tuning the backbone...', 'red'))
    if not finetune_backbone:
        for name, param in model.named_parameters():
            if 'projector' in name or 'predictor' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        params = list(filter(lambda p: p.requires_grad, model.parameters()))
    else:
        for name, param in model.named_parameters():
            if 'backbone' in name or 'projector' in name or 'predictor' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        if hasattr(model, "module"):
            backbone_params = list(
                filter(lambda p: p.requires_grad, model.module.backbone.parameters()))
            projector_params = list(
                filter(lambda p: p.requires_grad, model.module.projector.parameters()))
            predictor_params = list(
                filter(lambda p: p.requires_grad, model.module.predictor.parameters()))
        else:
            backbone_params = list(
                filter(lambda p: p.requires_grad, model.backbone.parameters()))
            projector_params = list(
                filter(lambda p: p.requires_grad, model.projector.parameters()))
            predictor_params = list(
                filter(lambda p: p.requires_grad, model.predictor.parameters()))

    if p['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(params, **p['optimizer_kwargs'])

    elif p['optimizer'] == 'adam':
        if not finetune_backbone or p['setup'] == 'simclr' or p['setup'] == 'moco':
            optimizer = torch.optim.Adam(params, **p['optimizer_kwargs'])
        else:
            if 'backbone_lr_scale' in p.keys():
                optimizer = torch.optim.Adam(
                    params=backbone_params, lr=p['optimizer_kwargs']['backbone_lr_scale'] *
                    p['optimizer_kwargs']['lr'],
                    weight_decay=p['optimizer_kwargs']['weight_decay'])
            else:
                optimizer = torch.optim.Adam(
                    params=backbone_params, lr=p['optimizer_kwargs']['lr'],
                    weight_decay=p['optimizer_kwargs']['weight_decay'])
            optimizer.add_param_group({
                "params": projector_params, "lr": p['optimizer_kwargs']['lr'],
                "weight_decay": p['optimizer_kwargs']['weight_decay']})
            optimizer.add_param_group({
                "params": predictor_params, "lr": p['optimizer_kwargs']['lr'],
                "weight_decay": p['optimizer_kwargs']['weight_decay']})
    else:
        raise ValueError('Invalid optimizer {}'.format(p['optimizer']))

    return optimizer


def adjust_learning_rate(p, optimizer, epoch):
    lr = p['optimizer_kwargs']['lr']

    if p['scheduler'] == 'cosine':
        eta_min = lr * (p['scheduler_kwargs']['lr_decay_rate'] ** 3)
        lr = eta_min + (lr - eta_min) * \
            (1 + math.cos(math.pi * epoch / p['epochs'])) / 2

    elif p['scheduler'] == 'step':
        steps = np.sum(epoch > np.array(
            p['scheduler_kwargs']['lr_decay_epochs']))
        if steps > 0:
            lr = lr * (p['scheduler_kwargs']['lr_decay_rate'] ** steps)

    elif p['scheduler'] == 'constant':
        lr = lr

    else:
        raise ValueError(
            'Invalid learning rate schedule {}'.format(p['scheduler']))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def collate_custom(batch):
    if isinstance(batch[0], np.int64):
        return np.stack(batch, 0)

    if isinstance(batch[0], torch.Tensor):
        return torch.stack(batch, 0)

    elif isinstance(batch[0], np.ndarray):
        return np.stack(batch, 0)

    elif isinstance(batch[0], int):
        return torch.LongTensor(batch)

    elif isinstance(batch[0], float):
        return torch.FloatTensor(batch)

    elif isinstance(batch[0], str):
        return batch

    elif isinstance(batch[0], collections.Mapping):
        batch_modified = {key: collate_custom(
            [d[key] for d in batch]) for key in batch[0] if key.find('idx') < 0}
        return batch_modified

    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [collate_custom(samples) for samples in transposed]

    raise TypeError(('Type is {}'.format(type(batch[0]))))
