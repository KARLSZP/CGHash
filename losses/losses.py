"""
    Part of Codes are forked from other work(s).
    Links and Reference would be added in open-source version.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
EPS = 1e-8


""" CGHash """


def entropy(x, input_as_probabilities):
    """
    Helper function to compute the entropy over the batch

    input: batch w/ shape [b, num_classes]
    output: entropy value [is ideally -log(num_classes)]
    """

    if input_as_probabilities:
        x_ = torch.clamp(x, min=EPS)
        b = x_ * torch.log(x_)
    else:
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)

    if len(b.size()) == 2:  # Sample-wise entropy
        return -b.sum(dim=1).mean()
    elif len(b.size()) == 1:  # Distribution-wise entropy
        return -b.sum()
    else:
        raise ValueError('Input tensor is %d-Dimensional' % (len(b.size())))


class ConsistentLoss(nn.Module):
    def __init__(self, consistent_w=1.0):
        super(ConsistentLoss, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.bce = nn.BCELoss()
        self.consistent_w = consistent_w

    def forward(self, anchors, neighbors):
        """
        input:
            - anchors: logits for anchor images w/ shape [b, num_classes]
            - neighbors: logits for neighbor images w/ shape [b, num_classes]

        output:
            - Loss
        """
        # Softmax
        b, n = anchors.size()
        anchors_prob = self.softmax(anchors)
        positives_prob = self.softmax(neighbors)

        # Similarity in pseudo-label space
        similarity = torch.bmm(anchors_prob.view(
            b, 1, n), positives_prob.view(b, n, 1)).squeeze()
        ones = torch.ones_like(similarity)
        consistency_loss = -ones*torch.log(similarity)
        consistency_loss = consistency_loss.mean()

        loss = self.consistent_w * consistency_loss
        return loss, consistency_loss.detach()


class EntropyLoss(nn.Module):
    def __init__(self, entropy_w=2.0, sharp_w=1.0):
        super(EntropyLoss, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.entropy_w = entropy_w
        self.sharp_w = sharp_w

    def forward(self, anchors):
        """
        input:
            - anchors: logits for anchor images w/ shape [b, num_classes]
            - neighbors: logits for neighbor images w/ shape [b, num_classes]

        output:
            - Loss
        """
        # Softmax
        anchors_prob = self.softmax(anchors)

        # Entropy loss
        entropy_loss = -entropy(torch.mean(anchors_prob, 0),
                                input_as_probabilities=True)
        sharp_loss = entropy(anchors_prob, input_as_probabilities=True)

        loss = self.entropy_w * entropy_loss + self.sharp_w * sharp_loss

        return loss, loss.detach()


class MaskedContLoss(nn.Module):
    def __init__(self, w, tau=1.0, pos_reweight=None):
        super(MaskedContLoss, self).__init__()
        self.w = w
        self.tau = tau
        self.pos_reweight = pos_reweight

    def cos_sim(self, a, b):
        if a.shape == b.shape:
            sim = (a * b).sum(dim=1) / a.shape[1]
        elif a.shape == b.T.shape:
            sim = (a @ b) / a.shape[1]
        else:
            raise ValueError("Invalid input shape.")

        return sim

    def forward(self, anchors_code, neighbors_code, pos_mask):
        """
        input:
            - anchors_code: codes for anchor images w/ shape [b, code_length]
            - neighbors_code: codes for neighbor images w/ shape [b, code_length]
            - negatives_code: codes for negative images w/ shape [b, code_length]
            - pos_mask: mask of confident possible neighbors w/ shape [b, b]
            - neg_mask: mask of confident possible negatives w/ shape [b, b]
        output:
            - Loss
        """
        # feature term
        pos_sim = self.cos_sim(anchors_code, neighbors_code) / self.tau  # b,
        neg_sim = self.cos_sim(anchors_code, anchors_code.T) / self.tau  # b,
        proto_code = pos_mask.float() @ anchors_code / \
            (pos_mask.sum(dim=1)[:, None]+EPS)  # b, L
        proto_pos_sim = self.cos_sim(
            neighbors_code, proto_code) / self.tau  # b,
        proto_neg_sim = self.cos_sim(anchors_code, proto_code) / self.tau  # b,

        if self.pos_reweight is not None:
            pos_term = self.pos_reweight * pos_sim.exp() +\
                (1 - self.pos_reweight) * \
                proto_pos_sim.exp()  # b, (+EPS to avoid zero division)
            neg_term = self.pos_reweight * (neg_sim.exp().sum(dim=1) - neg_sim.exp().diag()) +\
                (1 - self.pos_reweight) * proto_neg_sim.exp()
        else:
            pos_term = (pos_sim + proto_pos_sim) / 2  # b,

        cont = -torch.log(pos_term / (pos_term + neg_term)).mean()
        loss = self.w * cont

        return loss, cont.detach()


""" Self-Labeling """


class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(MaskedCrossEntropyLoss, self).__init__()

    def forward(self, input, target, mask, weight, reduction='mean'):
        if not (mask != 0).any():
            raise ValueError('Mask in MaskedCrossEntropyLoss is all zeros.')
        target = torch.masked_select(target, mask)
        b, c = input.size()
        n = target.size(0)
        input = torch.masked_select(input, mask.view(b, 1)).view(n, c)
        return F.cross_entropy(input, target, weight=weight, reduction=reduction)


class ConfidenceBasedCE(nn.Module):
    def __init__(self, threshold, threshold_max, apply_class_balancing, w=1.0):
        super(ConfidenceBasedCE, self).__init__()
        self.loss = MaskedCrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)
        self.threshold = threshold
        self.threshold_min = threshold
        self.threshold_max = threshold_max
        self.selected_rate = []
        self.apply_class_balancing = apply_class_balancing

    def adjust_threshold(self, epoch, num_epochs):
        num_epochs = num_epochs // 2
        if epoch <= num_epochs:
            self.threshold = self.threshold_min + 1. * epoch / \
                num_epochs * (self.threshold_max - self.threshold_min)
        return self.threshold

    def check_selected_rate(self):
        mean_selected_rate = sum(self.selected_rate) / len(self.selected_rate)
        self.selected_rate = []
        return mean_selected_rate

    def forward(self, anchors_weak, anchors_strong):
        """
        Loss function during self-labeling

        input: logits for original samples and for its strong augmentations
        output: cross entropy
        """
        # Retrieve target and mask based on weakly augmentated anchors
        weak_anchors_prob = self.softmax(anchors_weak)
        max_prob, target = torch.max(weak_anchors_prob, dim=1)

        # topk threshold
        # threshold = max_prob.topk(k=int(len(max_prob)*self.threshold))[0][-1]
        # mask = max_prob > threshold
        # conf threshold
        mask = max_prob > self.threshold

        self.selected_rate.append(mask.sum() / len(mask))
        # input()
        b, c = weak_anchors_prob.size()
        target_masked = torch.masked_select(target, mask.squeeze())
        n = target_masked.size(0)

        # Inputs are strongly augmented anchors
        input_ = anchors_strong

        # Class balancing weights
        if self.apply_class_balancing:
            idx, counts = torch.unique(target_masked, return_counts=True)
            freq = 1/(counts.float()/n)
            weight = torch.ones(c).cuda()
            weight[idx] = freq

        else:
            weight = None

        # Loss
        loss = self.loss(input_, target, mask, weight=weight, reduction='mean')

        return loss
