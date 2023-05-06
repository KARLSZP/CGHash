import argparse
import os
import random

import numpy as np
import torch
from termcolor import colored
from datetime import datetime

from utils.common_config import (adjust_learning_rate, get_base_dataloader,
                                 get_base_dataset, get_criterion, get_model,
                                 get_optimizer, get_train_dataloader,
                                 get_train_dataset, get_train_transformations,
                                 get_val_dataloader, get_val_dataset,
                                 get_val_transformations)
from utils.config import create_config
from utils.evaluate_utils import (evaluate_hash, evaluate_pl)
from utils.cghash_train import cghash_train


MAX_BAD_EPOCHS = 4
# Reproducibility
seed = 8888
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
torch.use_deterministic_algorithms(True)
torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser(description='cgHash training script')
parser.add_argument('--config_env', help='Location of path config file')
parser.add_argument('--config_exp', help='Location of experiments config file')
parser.add_argument('--gpu_id', '-g', help='Id of GPUs')
parser.add_argument('--test', '-t', action='store_true', help='Test mode?')
parser.add_argument('--force', '-f', action='store_true', help='Ignore ckpts?')
parser.add_argument('--wait', '-w', help='waiting time')
args = parser.parse_args()

if args.gpu_id is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id


def main():
    # load config
    p = create_config(args.config_env, args.config_exp)

    if args.wait is not None:
        os.system("sleep {}".format(args.wait))

    # model
    print(colored('Load model', 'blue'))
    model = get_model(p, p['pretext_model'])
    model = torch.nn.DataParallel(model)
    model = model.cuda()

    # optimizer
    print(colored('Load optimizer', 'blue'))
    optimizer = get_optimizer(p, model, p['finetune_backbone'])

    # transforms
    train_transformations = get_train_transformations(p)
    val_transformations = get_val_transformations(p)

    # datasets
    print(colored('Load dataset and dataloaders', 'blue'))
    base_dataset = get_base_dataset(p, val_transformations)
    train_dataset = get_train_dataset(
        p, train_transformations, to_neighbors_dataset=True)
    val_dataset = get_val_dataset(p, val_transformations)

    # dataloaders
    base_dataloader = get_base_dataloader(p, base_dataset)
    train_dataloader = get_train_dataloader(p, train_dataset)
    val_dataloader = get_val_dataloader(p, val_dataset)
    print('Train samples %d - Val samples %d' %
          (len(train_dataset), len(val_dataset)))

    # loss function
    print(colored('Load loss functions', 'blue'))
    criterion, criterion_state = get_criterion(p)
    for cri in criterion.values():
        cri.cuda()

    # checkpoint
    if (not args.force) and os.path.exists(p['cghash_checkpoint']):
        print(colored('Restart from checkpoint {}'.format(
            p['cghash_checkpoint']), 'blue'))
        checkpoint = torch.load(p['cghash_checkpoint'], map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        bad_epochs = checkpoint['bad_epochs']
        best_mAP = checkpoint['best_mAP']

    else:
        print(colored('No checkpoint file at {}'.format(
            p['cghash_checkpoint']), 'blue'))
        start_epoch = 0
        bad_epochs = 0
        best_mAP = 0

    # settings
    MULTI = True if p['train_db_name'] in ['coco', 'nus'] else False

    # Test
    if args.test:
        print(colored('Loading Testing Model {}'.format(
            p['cghash_model']), 'blue'))
        model_checkpoint = torch.load(p['cghash_model'], map_location='cpu')
        model.module.load_state_dict(model_checkpoint['model'])

        print(' | evaluating Pseudo-label Accuracy....')
        pred_dict, pred_stats = evaluate_pl(p, val_dataloader, model, MULTI)
        print(", ".join(["{}: {}".format(k, v) for k, v in sorted(
            pred_dict.items(), key=lambda x: x[0])]))
        print(colored(''.join([" | {}: {:.4f}".format(k, v)
              for k, v in pred_stats.items()]), 'yellow'))

        # Evaluate hashing perf.
        print(' | evaluating hash perf....')
        evals = evaluate_hash(model, base_dataloader, val_dataloader,
                              p['topk'], multilabel=MULTI)
        print(colored(''.join([" | {}: {:.4f}".format(k, v)
              for k, v in evals.items()]), 'yellow'))
        exit()

    """ ======================== Training Begins ======================== """
    # Main loop
    print(colored('Starting main loop', 'blue'))
    try:
        for epoch in range(start_epoch, p['epochs']):
            print(colored('-'*20, 'yellow'))
            print(colored('[{}] Epoch {}/{} - {}: {} - {}'.format(p['setup'], epoch+1,
                  p['epochs'], p['train_db_name'], p['cghash_subdir'], args.gpu_id), 'yellow'))

            # Adjust lr
            lr = adjust_learning_rate(p, optimizer, epoch)
            if p['scheduler'] != "constant":
                print(' | Adjusted learning rate to {:.5f}'.format(lr))

            # Train
            cghash_train(train_dataloader, model, criterion, criterion_state,
                         optimizer, epoch, p['finetune_backbone'])

            # Verbose
            if (epoch + 1) % p['eval_interval'] == 0:
                print(' | evaluating Pseudo-label Accuracy....')
                pred_dict, pred_stats = evaluate_pl(
                    p, val_dataloader, model, MULTI)
                print(", ".join(["{}: {}".format(k, v) for k, v in sorted(
                    pred_dict.items(), key=lambda x: x[0])]))
                print(colored(''.join([" | {}: {:.4f}".format(k, v)
                                       for k, v in pred_stats.items()]), 'yellow'))

                print(' | evaluating hash perf....')
                evals = evaluate_hash(model, base_dataloader,
                                      val_dataloader, p['topk'], multilabel=MULTI)
                print(colored(''.join([" | {}: {:.4f}".format(k, v)
                                       for k, v in evals.items()]), 'blue'))

                if evals['mAP'] > best_mAP:
                    bad_epochs = 0
                    print(colored(' | New best mAP found: {:.4f} > {:.4f}'.format(
                        best_mAP, evals['mAP']), "green"))
                    best_mAP = evals['mAP']
                    torch.save({'model': model.module.state_dict(),
                                'best_mAP': best_mAP},
                               p['cghash_model'])
                else:
                    bad_epochs += 1
                    print(colored(' | Not better than best mAP: {:.4f} < {:.4f} ({} bad epochs)'.format(
                        evals['mAP'], best_mAP, bad_epochs), "red"))

            # Save checkpoint
            torch.save({'optimizer': optimizer.state_dict(),
                        'model': model.state_dict(),
                        'epoch': epoch + 1, 'bad_epochs': bad_epochs,
                        'best_mAP': best_mAP}, p['cghash_checkpoint'])

            if bad_epochs > MAX_BAD_EPOCHS:
                print(" | Reach maximum bad epochs...")
                break
    except KeyboardInterrupt:
        print('-' * 89)
        print(' | Exiting from training...')

    """ ======================== Training Accomplished ======================== """
    # Evaluate the final model
    print(colored('Evaluate best model at the end', 'blue'))
    model_checkpoint = torch.load(p['cghash_model'], map_location='cpu')
    model.module.load_state_dict(model_checkpoint['model'])

    print(' | evaluating Pseudo-label Accuracy....')
    pred_dict, clustering_stats = evaluate_pl(
        p, val_dataloader, model, MULTI)
    print(colored(''.join([" | {}: {:.4f}".format(k, v)
                           for k, v in clustering_stats.items()]), 'yellow'))

    print(' | evaluating hash perf....')
    evals = evaluate_hash(model, base_dataloader,
                          val_dataloader, p['topk'], multilabel=MULTI)
    print(colored(''.join([" | {}: {:.4f}".format(k, v)
          for k, v in evals.items()]), 'yellow'))

    # Logging
    now = datetime.now()
    dt_string = now.strftime("%Y/%m/%d %H:%M:%S")
    with open("results/{}/best_map.log".format(p['train_db_name']), "a+") as f:
        f.write("{} {}: {:.4f}\n".format(
            dt_string, p['cghash_subdir'], best_mAP))

    print(" | CGHash: {}".format(p['cghash_subdir']))


if __name__ == "__main__":
    main()
