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
from utils.selflabel_train import selflabel_train


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
# torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
torch.use_deterministic_algorithms(True)


# Parser
parser = argparse.ArgumentParser(description='Self-labeling')
parser.add_argument('--config_env', help='Config file for the environment')
parser.add_argument('--config_exp', help='Config file for the experiment')
parser.add_argument('--gpu_id', '-g', help='Id of GPUs')
parser.add_argument('--wait', '-w', help='waiting time')
parser.add_argument('--test', '-t', action='store_true',
                    help='Run in test mode?')
parser.add_argument('--force', '-f', action='store_true',
                    help='Ignore ckpts?')
args = parser.parse_args()

if args.gpu_id is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id


def main():
    # load config
    p = create_config(args.config_env, args.config_exp)

    if args.wait is not None:
        os.system("sleep {}".format(args.wait))

    # model
    print(colored('Retrieve model', 'blue'))
    model = get_model(p, p['cghash_model'])
    model = torch.nn.DataParallel(model)
    model = model.cuda()

    # optimizer
    print(colored('Retrieve optimizer', 'blue'))
    optimizer = get_optimizer(p, model, p['finetune_backbone'])

    # transforms
    strong_transforms = get_train_transformations(p)
    val_transforms = get_val_transformations(p)

    # datasets
    print(colored('Load dataset and dataloaders', 'blue'))
    train_dataset = get_train_dataset(
        p, {'standard': val_transforms, 'augment': strong_transforms}, to_augmented_dataset=True)
    val_dataset = get_val_dataset(p, val_transforms)
    base_dataset = get_base_dataset(p, val_transforms)

    # dataloaders
    train_dataloader = get_train_dataloader(p, train_dataset)
    val_dataloader = get_val_dataloader(p, val_dataset)
    base_dataloader = get_base_dataloader(p, base_dataset)
    print(colored('Train samples %d - Val samples %d' %
                  (len(train_dataset), len(val_dataset)), 'yellow'))

    # loss function
    print(colored('Load loss functions', 'blue'))
    criterion, criterion_state = get_criterion(p)
    for cri in criterion.values():
        cri.cuda()

    # checkpoint
    if (not args.force) and os.path.exists(p['selflabel_checkpoint']):
        print(colored('Restart from checkpoint {}'.format(
            p['selflabel_checkpoint']), 'blue'))
        checkpoint = torch.load(p['selflabel_checkpoint'], map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        bad_epochs = checkpoint['bad_epochs']
        best_map = checkpoint['best_map']
    else:
        print(colored('No checkpoint file at {}'.format(
            p['selflabel_checkpoint']), 'blue'))
        start_epoch = 0
        bad_epochs = 0
        best_map = 0

    # settings
    MULTI = True if p['train_db_name'] in ['coco', 'nus'] else False

    # Test
    if args.test:
        model_checkpoint = torch.load(p['selflabel_model'], map_location='cpu')
        model.module.load_state_dict(model_checkpoint)

        print(' | evaluating Pseudo-label Accuracy....')
        pred_dict, pred_stats = evaluate_pl(p, val_dataloader, model, MULTI)
        print(", ".join(["{}: {}".format(k, v) for k, v in sorted(
            pred_dict.items(), key=lambda x: x[0])]))
        print(colored(''.join([" | {}: {:.4f}".format(k, v)
              for k, v in pred_stats.items()]), 'yellow'))

        print(' | evaluating hash perf....')
        evals = evaluate_hash(model, base_dataloader,
                              val_dataloader, p['topk'], multilabel=MULTI)
        print(colored(''.join([" | {}: {:.4f}".format(k, v)
                               for k, v in evals.items()]), 'blue'))
        exit()

    """ ======================== Training Begins ======================== """
    # Main loop
    print(colored('Starting main loop...', 'blue'))
    try:
        for epoch in range(start_epoch, p['epochs']):
            print(colored('-'*20, 'yellow'))
            print(colored('[{}] Epoch {}/{} - {}: {} - {}'.format(p['setup'], epoch+1,
                  p['epochs'], p['train_db_name'], p['selflabel_subdir'], args.gpu_id), 'yellow'))

            # Adjust lr
            lr = adjust_learning_rate(p, optimizer, epoch)
            if p['scheduler'] != "constant":
                print(' | Adjusted learning rate to {:.5f}'.format(lr))

            # Train
            selflabel_train(train_dataloader, model, criterion,
                            criterion_state, optimizer, epoch)
            selected_rate = criterion['confidenceCE'].check_selected_rate()
            print(' | Selected Rate {:.5f}'.format(selected_rate))

            # Verbose
            if (epoch + 1) % p['eval_interval'] == 0:
                print(' | evaluating Pseudo-label Accuracy....')
                pred_dict, clustering_stats = evaluate_pl(
                    p, val_dataloader, model, MULTI)
                print(", ".join(["{}: {}".format(k, v) for k, v in sorted(
                    pred_dict.items(), key=lambda x: x[0])]))

                print(colored(''.join([" | {}: {:.4f}".format(k, v)
                                       for k, v in clustering_stats.items()]), 'yellow'))

                print(' | evaluating hash perf....')
                evals = evaluate_hash(model, base_dataloader, val_dataloader,
                                      p['topk'], multilabel=MULTI)
                print(colored(''.join([" | {}: {:.4f}".format(k, v)
                                       for k, v in evals.items()]), 'blue'))

                if evals['mAP'] > best_map:
                    bad_epochs = 0
                    print(colored(' | New best mAP found: {:.4f} > {:.4f}'.format(
                        best_map, evals['mAP']), "green"))
                    best_map = evals['mAP']
                    torch.save(model.module.state_dict(), p['selflabel_model'])
                else:
                    bad_epochs += 1
                    print(colored(' | Not better than best mAP: {:.4f} < {:.4f} ({} bad epochs)'.format(
                        evals['mAP'], best_map, bad_epochs), "red"))

            # Save checkpoint
            torch.save({'optimizer': optimizer.state_dict(),
                        'model': model.state_dict(),
                        'epoch': epoch + 1, 'bad_epochs': bad_epochs,
                        'best_map': best_map}, p['selflabel_checkpoint'])

            if bad_epochs > MAX_BAD_EPOCHS:
                print(" | Reach maximum bad epochs...")
                break
    except KeyboardInterrupt:
        print('-' * 89)
        print(' | Exiting from training...')

    """ ======================== Training Accomplished ======================== """
    # Evaluate the final model
    print(colored('Evaluate model at the end', 'blue'))
    model_checkpoint = torch.load(p['selflabel_model'], map_location='cpu')
    model.module.load_state_dict(model_checkpoint)

    print(' | evaluating Pseudo-label Accuracy....')
    pred_dict, clustering_stats = evaluate_pl(
        p, val_dataloader, model, MULTI)
    print(colored(''.join([" | {}: {:.4f}".format(k, v)
                           for k, v in clustering_stats.items()]), 'yellow'))

    # Evaluate hashing perf.
    print(' | evaluating hash perf....')
    evals = evaluate_hash(model.module, base_dataloader,
                          val_dataloader, p['topk'], multilabel=MULTI)
    print(colored(''.join([" | {}: {:.4f}".format(k, v)
                           for k, v in evals.items()]), 'blue'))

    now = datetime.now()
    dt_string = now.strftime("%Y/%m/%d %H:%M:%S")
    with open("results/{}/best_map.log".format(p['train_db_name']), "a+") as f:
        f.write("{} {}: {}\n".format(
            dt_string, p['selflabel_subdir'], best_map))

    print(" | CGHash: {}".format(p['cghash_subdir']))
    print(" | Self-Labeling: {}".format(p['selflabel_subdir']))


if __name__ == "__main__":
    main()
