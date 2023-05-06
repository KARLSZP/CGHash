import torch
from utils.utils import AverageMeter, ProgressMeter
from tqdm import tqdm


def cghash_train(train_loader, model, criterion, criterion_state,
                 optimizer, epoch, finetune_backbone=True):
    total_losses = AverageMeter('Total Loss', ':.4f')
    losses_dict = {}
    for k in criterion:
        losses_dict[k] = AverageMeter(k, ':.4f')

    progress = ProgressMeter(len(train_loader),
                             [total_losses, *losses_dict.values()],
                             prefix="Epoch: [{}]".format(epoch))

    if not finetune_backbone:
        model.module.backbone.eval()
    else:
        model.train()  # Update BN

    use_amp = True
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        x = batch['anchor'].cuda(non_blocking=True)
        x_feat = batch['neighbor'].cuda(non_blocking=True)

        if finetune_backbone:
            x_code, x_pred = model(x)
            x_feat_code, x_feat_pred = model(x_feat)
        else:
            with torch.no_grad():
                x_features = model(x, channel='backbone')
                x_feat_features = model(x_feat, channel='backbone')

            x_code, x_pred = model(x_features, channel='head')
            x_feat_code, x_feat_pred = model(x_feat_features, channel='head')

        """ Loss Computation """
        """  -- Loss in account """
        total_loss = torch.tensor(0.0).cuda()
        # Consitent Loss & Inconsistent Loss
        if criterion_state['consist']:
            consistent_loss, consistent = criterion['consist'](
                x_pred, x_feat_pred)
            total_loss += consistent_loss

        # MI Loss
        if criterion_state['MI']:
            MI_loss, MI = criterion['MI'](x_pred)
            total_loss += MI_loss

        if criterion_state['mcont']:
            prob = torch.softmax(x_pred, dim=1)
            max_prob, target = torch.max(prob, dim=1)
            pos_mask = (target.unsqueeze(1) == target).fill_diagonal_(False)

            mcont_loss, mcont = criterion['mcont'](
                x_code, x_feat_code, pos_mask)

            total_loss += mcont_loss

        """ Loss Verbose """
        # Register the mean loss and backprop the total loss to cover all subheads
        total_losses.update(total_loss)
        losses_dict['consist'].update(consistent)
        losses_dict['MI'].update(MI)
        losses_dict['mcont'].update(mcont)

        """ Back propagation """
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

    progress.display_end()
