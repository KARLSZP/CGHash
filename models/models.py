import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class ContrastiveModel(nn.Module):
    def __init__(self, backbone, head='mlp', features_dim=128):
        super(ContrastiveModel, self).__init__()
        self.backbone = backbone['backbone']
        self.backbone_dim = backbone['dim']
        self.head = head

        if head == 'linear':
            self.contrastive_head = nn.Linear(self.backbone_dim, features_dim)

        elif head == 'mlp':
            self.contrastive_head = nn.Sequential(
                nn.Linear(self.backbone_dim, self.backbone_dim),
                nn.ReLU(inplace=True), nn.Linear(self.backbone_dim, features_dim))

        else:
            raise ValueError('Invalid head {}'.format(head))

    def forward(self, x):
        features = self.contrastive_head(self.backbone(x))
        features = F.normalize(features, dim=1)
        return features


class CGHashModel(nn.Module):
    def __init__(self, backbone, nclusters, encode_length, tau):
        super(CGHashModel, self).__init__()
        self.backbone = backbone['backbone']
        self.backbone_dim = backbone['dim']
        self.projector = nn.Sequential(
            nn.Linear(self.backbone_dim, self.backbone_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(self.backbone_dim // 4, encode_length),
        )
        self.predictor = nn.Linear(encode_length, nclusters)
        self.tau = tau

    def forward(self, x, channel='default'):
        if channel == 'default':
            features = self.backbone(x)
            proj = self.projector(features)
            prob = torch.sigmoid(proj / self.tau)
            u = torch.empty_like(prob).uniform_().to(prob.device)
            z = hash_layer(prob - u)
            out = self.predictor(z)
            out = (z, out)

        elif channel == 'backbone':
            out = self.backbone(x)

        elif channel == 'head':
            proj = self.projector(x)
            prob = torch.sigmoid(proj / self.tau)
            u = torch.empty_like(prob).uniform_().to(prob.device)
            z = hash_layer(prob - u)
            out = self.predictor(z)
            out = (z, out)

        # hash evaluation, raw features
        elif channel == 'encode':
            features = self.backbone(x)
            proj = self.projector(features)
            out = proj

        # pseudo-label evaluation
        elif channel == 'all':
            features = self.backbone(x)
            proj = self.projector(features)
            prob = torch.sigmoid(proj / self.tau)
            u = 0.5
            z = hash_layer(prob - u)
            out = {'features': features,
                   'z': z,
                   'output': self.predictor(z)}

        else:
            raise ValueError('Invalid forward pass {}'.format(channel))

        return out


class hash(Function):
    @ staticmethod
    def forward(ctx, input_):
        return torch.sign(input_)

    @ staticmethod
    def backward(ctx, grad_output):
        return grad_output


def hash_layer(input):
    return hash.apply(input)
