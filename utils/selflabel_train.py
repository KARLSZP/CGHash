import torch
import numpy as np
from utils.utils import AverageMeter, ProgressMeter
from tqdm import tqdm


def selflabel_train(train_loader, model, criterion, criterion_state, optimizer, epoch):
    """
        Self-labeling based on confident samples
    """
    total_losses = AverageMeter('Total Loss', ':.4f')
    losses_dict = {}
    for k in criterion:
        losses_dict[k] = AverageMeter(k, ':.4f')
    progress = ProgressMeter(len(train_loader), [total_losses, *losses_dict.values()],
                             prefix="Epoch: [{}]".format(epoch))
    model.train()

    use_amp = True
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    for i, batch in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        images = batch['image'].cuda(non_blocking=True)
        images_augmented = batch['image_augmented'].cuda(non_blocking=True)

        with torch.no_grad():
            _, output = model(images)
        _, output_augmented = model(images_augmented)

        """ Loss Computation """
        """  -- Loss in account """
        total_loss = torch.tensor(0.0).cuda()
        # confidenceCE Loss
        if criterion_state['confidenceCE']:
            selflabel_loss = criterion['confidenceCE'](
                output, output_augmented)
            total_loss += selflabel_loss

        """ Loss Verbose """
        total_losses.update(total_loss)
        losses_dict['confidenceCE'].update(selflabel_loss.detach())

        """ Back propagation """
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

    progress.display_end()
