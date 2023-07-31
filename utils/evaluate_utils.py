from collections import defaultdict

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from data.custom_dataset import NeighborsDataset
from scipy.optimize import linear_sum_assignment
from sklearn import metrics
from torch.autograd import Variable
from tqdm import tqdm
from utils.configurations import get_feature_dimensions_backbone


@torch.no_grad()
def get_predictions(p, dataloader, model, multilabel=False, return_features=False):
    # Make predictions on a dataset with neighbors
    model.eval()
    predictions = []
    probs = []
    targets = []
    if return_features:
        ft_dim = get_feature_dimensions_backbone(p)
        features = torch.zeros((len(dataloader.sampler), ft_dim)).cuda()

    if isinstance(dataloader.dataset, NeighborsDataset):  # Also return the neighbors
        key_ = 'anchor'
        include_neighbors = True
        neighbors = []

    else:
        key_ = 'image'
        include_neighbors = False

    ptr = 0
    for batch in dataloader:
        images = batch[key_].cuda(non_blocking=True)
        bs = images.shape[0]
        res = model(images, channel='all')
        output = res['output']
        if return_features:
            features[ptr: ptr+bs] = res['features']
            ptr += bs
        predictions.append(torch.argmax(output, dim=1))
        probs.append(F.softmax(output, dim=1))
        if multilabel:
            targets.append(batch['multi_target'])
        else:
            targets.append(batch['target'])
        if include_neighbors:
            neighbors.append(batch['possible_neighbors'])

    predictions = torch.cat(predictions, dim=0).cpu()
    probs = torch.cat(probs, dim=0).cpu()
    targets = torch.cat(targets, dim=0)

    if include_neighbors:
        neighbors = torch.cat(neighbors, dim=0)
        out = {'predictions': predictions, 'probabilities': probs,
               'targets': targets, 'neighbors': neighbors}

    else:
        out = {'predictions': predictions,
               'probabilities': probs, 'targets': targets}

    if return_features:
        return out, features.cpu()
    else:
        return out


@torch.no_grad()
def hungarian_evaluate(output, multilabel=False):
    # Evaluate model based on hungarian matching between predicted cluster assignment and gt classes.
    # This is computed only for the passed subhead index.

    # Hungarian matching
    targets = output['targets'].cuda()
    predictions = output['predictions'].cuda()
    probs = output['probabilities'].cuda()
    if multilabel:
        num_classes = targets.shape[1]
    else:
        num_classes = torch.unique(targets).numel()
    num_elems = targets.size(0)
    match = _hungarian_match(predictions, targets,
                             preds_k=num_classes, targets_k=num_classes)
    reordered_preds = torch.zeros(num_elems, dtype=predictions.dtype).cuda()
    for pred_i, target_i in match:
        reordered_preds[predictions == int(pred_i)] = int(target_i)

    if multilabel:
        reordered_preds = F.one_hot(reordered_preds, targets.shape[1])
        acc = int(((reordered_preds * targets).sum(dim=1)
                  > 0).sum()) / float(num_elems)
        ari = np.NaN
        nmi = np.NaN
        top5 = np.NaN

    else:
        # Gather performance metrics
        acc = int((reordered_preds == targets).sum()) / float(num_elems)
        nmi = metrics.normalized_mutual_info_score(
            targets.cpu().numpy(), predictions.cpu().numpy())
        ari = metrics.adjusted_rand_score(
            targets.cpu().numpy(), predictions.cpu().numpy())

        _, preds_top5 = probs.topk(5, 1, largest=True)
        reordered_preds_top5 = torch.zeros_like(preds_top5)
        for pred_i, target_i in match:
            reordered_preds_top5[preds_top5 == int(pred_i)] = int(target_i)
        correct_top5_binary = reordered_preds_top5.eq(
            targets.view(-1, 1).expand_as(reordered_preds_top5))
        top5 = float(correct_top5_binary.sum()) / float(num_elems)

    return {'ACC': acc, 'ARI': ari, 'NMI': nmi, 'ACC Top-5': top5}, match
    # return {'ACC': acc, 'ARI': ari, 'NMI': nmi, 'ACC Top-5': top5, 'hungarian_match': match}


@torch.no_grad()
def _hungarian_match(flat_preds, flat_targets, preds_k, targets_k):
    # Based on implementation from IIC
    num_samples = flat_targets.shape[0]

    assert (preds_k == targets_k)  # one to one
    num_k = preds_k
    num_correct = np.zeros((num_k, num_k))

    for c1 in range(num_k):
        for c2 in range(num_k):
            if flat_targets.dim() > 1:
                # elementwise, so each sample contributes once
                votes = int(
                    ((flat_preds == c1) * (flat_targets[:, c2] == 1)).sum())
                num_correct[c1, c2] = votes
            else:
                # elementwise, so each sample contributes once
                votes = int(((flat_preds == c1) * (flat_targets == c2)).sum())
                num_correct[c1, c2] = votes

    # num_correct is small
    match = linear_sum_assignment(num_samples - num_correct)
    match = np.array(list(zip(*match)))

    # return as list of tuples, out_c to gt_c
    res = []
    for out_c, gt_c in match:
        res.append((out_c, gt_c))

    return res


def evaluate_hash(model, database_loader, val_loader, topk, multilabel=False, cal_pr=False):
    evals = {}
    if hasattr(model, "training"):
        is_training = model.training
    model.eval()
    with torch.no_grad():
        retrievalB, retrievalL, queryB, queryL, codes = compress(
            database_loader, val_loader, model,
            multilabel=multilabel)
        avg_ham = calculate_avg_hamming(codes)
        result = calculate_top_map(
            qB=queryB, rB=retrievalB, queryL=queryL, retrievalL=retrievalL,
            topk=topk, multilabel=multilabel)

    evals["mAP"] = result
    evals["avg_ham"] = avg_ham

    if cal_pr:
        evals["P-R"] = pr_curve(qB=queryB, rB=retrievalB, query_label=queryL,
                                retrieval_label=retrievalL, tqdm_label='')

    model.training = is_training
    return evals


def evaluate_pl(p, val_dataloader, model, multilabel):
    predictions = get_predictions(
        p, val_dataloader, model, multilabel=multilabel)
    clustering_stats, match = hungarian_evaluate(
        predictions, multilabel=multilabel)
    pred, counts = predictions['predictions'].unique(
        return_counts=True)
    reordered_preds = torch.zeros(len(pred), dtype=pred.dtype).cuda()
    for pred_i, target_i in match:
        reordered_preds[pred == int(pred_i)] = int(target_i)
    pred_dict = dict(
        zip(reordered_preds.cpu().tolist(), counts.tolist()))
    return pred_dict, clustering_stats


def compress(train, test, model, multilabel=False, device="cuda"):
    retrievalB = list([])
    retrievalL = list([])
    codes = defaultdict(list)
    queryB = list([])
    queryL = list([])
    for batch_step, batch in enumerate(tqdm(test)):
        var_data = Variable(batch['image'].to(device))
        if multilabel:
            target = batch['multi_target']
        else:
            target = batch['target']
        label = target
        code = model(var_data, channel='encode')

        if not multilabel:
            for i in range(len(code)):
                codes[int(label[i])].append(code[i].cpu().view(1, -1))
        queryB.extend(code.cpu().data.numpy())
        queryL.extend(target.clone())

    queryB = np.array(queryB)
    queryL = np.stack(queryL)

    if not multilabel:
        for k in codes:
            codes[k] = torch.cat(codes[k])

    for batch_step, batch in enumerate(tqdm(train)):
        var_data = Variable(batch['image'].to(device))
        if multilabel:
            target = batch['multi_target']
        else:
            target = batch['target']

        code = model(var_data, channel='encode')

        retrievalB.extend(code.cpu().data.numpy())
        retrievalL.extend(target.clone())

    retrievalB = np.array(retrievalB)
    retrievalL = np.stack(retrievalL)

    # medium sep binarilization
    q_sep = np.median(queryB, axis=0)
    r_sep = np.median(retrievalB, axis=0)

    # 0.5 binarilization
    # q_sep = np.median(queryB, axis=0)
    # r_sep = np.median(retrievalB, axis=0)

    queryB[queryB < q_sep] = -1.
    queryB[queryB != -1] = 1.

    if not multilabel:
        for k in codes:
            code = codes[k]
            code[code.numpy() < q_sep] = -1.
            code[code != -1] = 1.
            codes[k] = code

    retrievalB[retrievalB < r_sep] = -1.
    retrievalB[retrievalB != -1] = 1.

    if multilabel:
        del codes
        codes = None

    return retrievalB, retrievalL, queryB, queryL, codes


def calculate_hamming(B1, B2):
    """
    :param B1:  vector [n]
    :param B2:  vector [r*n]
    :return: hamming distance [r]
    """
    q = B2.shape[1]  # max inner product value
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH


def calculate_avg_hamming(codes):
    if codes is None:
        return np.NaN
    res = []
    cross_res = []
    min_cross_res = []
    print_strs = {}
    for k, v in codes.items():
        v_cuda = v.cuda()
        res.append(float(torch.pdist(v_cuda, p=0).mean().cpu()))
        cross_sum = []
        for k2, t in codes.items():
            if k == k2:
                continue
            cross_sum.append(
                float(torch.cdist(v_cuda, t.cuda(), p=0).mean().cpu()))

        cross_res.append(sum(cross_sum) / len(cross_sum))
        min_cross_res.append(min(cross_sum))

        print_strs[k] = " {}: {:8.4f} | {:12.4f} | {:12.4f}".format(
            k, res[-1], min_cross_res[-1], cross_res[-1])

    print(" | (intra) <---> (min_inter) <---> (avg_inter)")
    for i in range(len(print_strs)):
        print(print_strs[i])

    return sum(res) / len(res)


def calculate_top_map(qB, rB, queryL, retrievalL, topk, multilabel):
    """
    :param qB: {-1,+1}^{mxq} query bits
    :param rB: {-1,+1}^{nxq} retrieval bits
    :param queryL: {0,1}^{mxl} query label
    :param retrievalL: {0,1}^{nxl} retrieval label
    :param topk:
    :return:
    """
    if not multilabel:
        n_values = np.max(queryL) + 1
        queryL = np.eye(n_values)[queryL]
        retrievalL = np.eye(n_values)[retrievalL]
    num_query = queryL.shape[0]
    topkmap = 0
    rB = rB.astype(int)
    qB = qB.astype(int)
    rB[rB == -1] = 0
    qB[qB == -1] = 0
    rB = np.packbits(rB, axis=1)
    qB = np.packbits(qB, axis=1)
    index = faiss.IndexBinaryFlat(rB.shape[1]*8)
    index.add(rB)

    _, ind = index.search(qB, topk)

    for iter in tqdm(range(num_query)):
        gnd = (np.dot(queryL[iter, :], retrievalL.T) > 0).astype(
            np.float32)  # (1 x L) @ (L x N) -> (1 x N)
        gnd = gnd[ind[iter]]  # reorder gnd

        tgnd = gnd[:topk]
        tsum = int(np.sum(tgnd))
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)
        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    return topkmap


def pr_curve(qB, rB, query_label, retrieval_label, tqdm_label=''):

    qB = torch.from_numpy(qB)
    rB = torch.from_numpy(rB)
    query_label = torch.from_numpy(query_label)
    retrieval_label = torch.from_numpy(retrieval_label)

    if tqdm_label != '':
        tqdm_label = 'PR-curve ' + tqdm_label

    num_query = qB.shape[0]
    num_bit = qB.shape[1]
    P = torch.zeros(num_query, num_bit + 1)
    R = torch.zeros(num_query, num_bit + 1)

    # for each sample from query calculate precision and recall
    for i in tqdm(range(num_query), desc=tqdm_label):
        if query_label.dim() > 1:
            gnd = (query_label[i].unsqueeze(0).mm(
                retrieval_label.t()) > 0).float().squeeze()
        else:
            gnd = (query_label[i] == retrieval_label).float().squeeze()
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calculate_hamming(qB[i, :].numpy(), rB.numpy())
        hamm = torch.from_numpy(hamm)
        tmp = (hamm <= torch.arange(0, num_bit + 1).reshape(-1,
               1).float().to(hamm.device)).float()
        total = tmp.sum(dim=-1)
        total = total + (total == 0).float() * 0.0001
        t = gnd * tmp
        count = t.sum(dim=-1)
        p = count / total  # TP / (TP + FP)
        r = count / tsum  # TP / (TP + FN)
        P[i] = p
        R[i] = r

    P = P.mean(dim=0)
    R = R.mean(dim=0)
    return P, R
