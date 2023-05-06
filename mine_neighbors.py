import argparse
import os

import numpy as np
import torch
from termcolor import colored

from utils.common_config import (get_model, get_train_dataset,
                                 get_val_dataloader, get_val_transformations)
from utils.config import create_config
from utils.memory_bank import MemoryBank

# Parser
parser = argparse.ArgumentParser()
parser.add_argument('--config_env', help='Config file for the environment')
parser.add_argument('--config_exp', help='Config file for the experiment')
parser.add_argument('--gpu_id', '-g', help='Id of GPUs')
args = parser.parse_args()

if args.gpu_id is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id


def main():
    # Retrieve config file
    p = create_config(args.config_env, args.config_exp)

    # Model
    print(colored('Retrieve model', 'blue'))
    model = get_model(p, p['pretrained_path'])
    model = torch.nn.DataParallel(model)
    model = model.cuda()

    torch.backends.cudnn.benchmark = True
    MULTI = True if p['train_db_name'] in ['coco', 'nus'] else False

    # Dataset
    print(colored('Retrieve dataset (Multi-Label: {})', 'blue').format(MULTI))
    val_transforms = get_val_transformations(p)
    base_dataset = get_train_dataset(p, val_transforms)
    base_dataloader = get_val_dataloader(p, base_dataset)

    # Memory Bank
    print(colored('Build MemoryBank', 'blue'))
    memory_bank_base = MemoryBank(
        len(base_dataset), p['output_dim'],
        p['num_classes'], p['temperature']).cuda()

    # Mine Neighbors
    print(colored('Fill Memorybank', 'blue'))
    if p['use_identity_head']:
        contrastive_head = model.module.contrastive_head
        model.module.contrastive_head = torch.nn.Identity()
    memory_bank_base.fill_memory_bank(base_dataloader, model, multilabel=MULTI)

    topk = p['topk']
    print('Mine the nearest neighbors (Top-{})'.format(topk))
    indices, acc = memory_bank_base.mine_nearest_neighbors(
        topk, multilabel=MULTI)
    print('Accuracy of Top-{} nearest neighbors on train set is {}'.format(
        topk, colored("{:.2f}".format(100*acc), 'green')))

    # Save files
    np.save(p['topk_neighbors_train_path'], indices)
    print('Neighbor-Index stored at {}'.format(
        colored(p['topk_neighbors_train_path'], 'green')))

    if p['use_identity_head']:
        model.module.contrastive_head = contrastive_head
    torch.save(model.state_dict(), p['pretext_model'])
    print('Backbone model stored at {}'.format(
        colored(p['pretext_model'], 'green')))


if __name__ == '__main__':
    main()
