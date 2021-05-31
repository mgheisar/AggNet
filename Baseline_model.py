# Baseline Model(Sum for aggregation and Sign() for binarization, without logiritic regression)
import torch
import torch.nn as nn
import torch.nn.functional as F
from Models_weights.resnet50_128 import resnet50_128
from Models_weights.senet50_128 import senet50_128
from Models_weights.resnet50_ft_dims_2048 import resnet50_ft
from Models_weights.senet50_ft_dims_2048 import senet50_ft
import numpy as np
import os
from sklearn import metrics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


# ROOT_DIR = '/nfs/nas4/marzieh/marzieh/VGG_Face2/exp'

# print(ROOT_DIR)
class SumPooling(nn.Module):
    def __init__(self, normalize_input=True, dim=128):
        super(SumPooling, self).__init__()
        self.normalize_input = normalize_input
        self.bn = nn.BatchNorm1d(dim)

    def forward(self, x):
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim
        set_vec = torch.sum(x, dim=2).squeeze()
        set_vec = self.bn(set_vec)
        set_vec = F.normalize(set_vec, p=2, dim=1)  # L2 normalize
        return set_vec


class Baseline(nn.Module):
    def __init__(self, base_model_architecture="resnet50_128"):
        super(Baseline, self).__init__()
        if base_model_architecture == "resnet50_128":
            self.base_model = resnet50_128(ROOT_DIR + '/Models_weights/resnet50_128.pth')
            dim = 128
        elif base_model_architecture == "senet50_128":
            self.base_model = senet50_128(ROOT_DIR + '/Models_weights/senet50_128.pth')
            dim = 128
        elif base_model_architecture == "resnet50_2048":
            self.base_model = resnet50_ft(ROOT_DIR + '/Models_weights/resnet50_ft_dims_2048.pth')
            dim = 2048
        elif base_model_architecture == "senet50_2048":
            self.base_model = senet50_ft(ROOT_DIR + '/Models_weights/senet50_ft_dims_2048.pth')
            dim = 2048
        self.sum_pooling = SumPooling()
        self.bn_x = nn.BatchNorm1d(dim, affine=False)

    def forward(self, x, m):
        x, x_pre_flatten = self.base_model(x)
        x = x.view(int(x.shape[0] / m), m, x.shape[1]).unsqueeze(-1)
        x = x.permute(0, 2, 1, 3)
        # x = F.normalize(x, p=2, dim=1).squeeze()  # L2 normalize
        v_set = self.sum_pooling(x)
        code_set = torch.sign(v_set + 1e-16)
        return v_set, code_set


def acc_authentication(model, H0_id, H0_data, target, n_classes, v_set, v_f, m_set, n_batch_verif):
    H0_id = torch.stack(H0_id).flatten(start_dim=0, end_dim=1)
    H0_data = torch.stack(H0_data).flatten(start_dim=0, end_dim=1)
    indices = np.where(np.in1d(H0_id, target))[0]
    temp = np.arange(len(H0_id))
    temp[indices] = -1
    temp = temp[temp >= 0]
    # classes_1 = np.random.choice(len(temp), n_classes, replace=False)
    classes_ = torch.randperm(len(temp))[:n_classes]
    # H0_id = H0_id[temp[classes_]]
    H0_data = H0_data[temp[classes_]]

    temp = n_classes // n_batch_verif
    v_f0 = []
    # model.eval()
    with torch.no_grad():
        for i in range(n_batch_verif):
            v_, code_f0 = model(H0_data[i * temp:(i + 1) * temp].to(device), m=1)
            v_f0.append(code_f0)  # single vector per query
    v_f0 = torch.stack(v_f0).flatten(start_dim=0, end_dim=1)
    H0_claimed_group_id = torch.randint(n_classes // m_set, (n_classes,)).numpy().astype(np.int)
    # D00_ = torch.mm(v_set[H0_claimed_group_id], v_f0.t())
    # D00 = torch.diag(D00_).cpu()
    Sim = torch.mm(F.normalize(v_set[H0_claimed_group_id], p=2, dim=1), F.normalize(v_f0, p=2, dim=1).t())
    D00_ = Sim
    # D00_ = logisticReg(Sim.unsqueeze(-1)).squeeze()
    D00 = torch.diag(F.sigmoid(D00_)).cpu()

    H1_group_id = np.repeat(np.arange(n_classes // m_set), m_set)
    # D11_ = torch.mm(v_set[H1_group_id], v_f.t())
    # D11 = torch.diag(D11_).cpu()
    Sim = torch.mm(F.normalize(v_set[H1_group_id], p=2, dim=1), F.normalize(v_f, p=2, dim=1).t())
    D11_ = Sim
    # D11_ = logisticReg(Sim.unsqueeze(-1)).squeeze()
    D11 = torch.diag(F.sigmoid(D11_)).cpu()
    D0 = np.sort(D00)[::-1]
    D1 = np.sort(D11)[::-1]

    Pfp = 0.01
    tau = D0[int(Pfp * n_classes)]
    Ptp01 = np.count_nonzero(D1 > tau) / n_classes
    Pfp = 0.05
    tau = D0[int(Pfp * n_classes)]
    Ptp05 = np.count_nonzero(D1 > tau) / n_classes
    Pfp = 0.1
    tau = D0[int(Pfp * n_classes)]
    Ptp1 = np.count_nonzero(D1 > tau) / n_classes

    tau = np.linspace(D1[0], D0[-1], 100)  # endpoint=True
    fpr = np.zeros(len(tau))
    tpr = np.zeros(len(tau))
    for kt in range(len(tau)):
        fpr[kt] = np.count_nonzero(D0 > tau[kt]) / n_classes
        tpr[kt] = np.count_nonzero(D1 > tau[kt]) / n_classes
    auc = metrics.auc(fpr, tpr)
    return Ptp01, Ptp05, Ptp1, auc


class Net(nn.Module):
    def __init__(self, model_type="resnet50_128"):
        super(Net, self).__init__()
        if model_type == "resnet50_128":
            self.base_model = resnet50_128(ROOT_DIR + '/Models_weights/resnet50_128.pth')
            self.encoder_dim = 128
        elif model_type == "senet50_128":
            self.base_model = senet50_128(ROOT_DIR + '/Models_weights/senet50_128.pth')
            self.encoder_dim = 128
        elif model_type == "resnet50_2048":
            self.base_model = resnet50_ft(ROOT_DIR + '/Models_weights/resnet50_ft_dims_2048.pth')
            self.encoder_dim = 2048
        elif model_type == "senet50_2048":
            self.base_model = senet50_ft(ROOT_DIR + '/Models_weights/senet50_ft_dims_2048.pth')
            self.encoder_dim = 2048

    def forward(self, x):
        x, x_pre_flatten = self.base_model(x)
        return x



