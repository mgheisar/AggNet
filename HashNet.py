import torch
import torch.nn as nn
import torch.nn.functional as F
from Models_weights.resnet50_128 import resnet50_128
from Models_weights.senet50_128 import senet50_128
from Models_weights.resnet50_ft_dims_2048 import resnet50_ft
from Models_weights.senet50_ft_dims_2048 import senet50_ft
import numpy as np
import os
import faiss
from sklearn.neighbors import NearestNeighbors
import h5py
from utils_data import BalanceBatchSampler
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT_DIR = '/nfs/nas4/marzieh/marzieh/VGG_Face2/exp'

# print(ROOT_DIR)


class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, num_clusters=8, dim=128, vlad_dim=128, vlad_v2=False,
                 normalize_input=True):
        """
        Args:
            num_clusters : int
                The number of clusters
            vlad_dim : int
                Dimension of final vlad vector
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
            vlad_v2 : bool
                If true, use vladv2 otherwise use vladv1
        """
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.vlad_dim = vlad_dim
        self.dim = dim
        self.vlad_v2 = vlad_v2
        self.alpha = 0
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(vlad_dim, num_clusters, kernel_size=(1, 1), bias=vlad_v2)
        self.centroids = nn.Parameter(torch.rand(num_clusters, vlad_dim))
        self.fc = nn.Linear(num_clusters*dim, vlad_dim)
        self.bn = nn.BatchNorm1d(num_clusters*dim)  # affine=False?,momentum=0.01?,vlad_dim if fc is applied
        # self._init_params()

    # def _init_params(self):
    #     self.conv.weight = nn.Parameter(
    #         (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
    #     )
    #     self.conv.bias = nn.Parameter(
    #         - self.alpha * self.centroids.norm(dim=1)
    #     )
    def init_params(self, clsts, traindescs):
        # TODO replace numpy ops with pytorch ops
        if not self.vlad_v2:
            clsts_ = clsts / np.linalg.norm(clsts, axis=1, keepdims=True)
            traindescs_ = traindescs / np.linalg.norm(traindescs, axis=1, keepdims=True)
            dots = np.dot(clsts_, traindescs_.T)
            dots.sort(0)
            dots = dots[::-1, :]  # sort, descending
            del traindescs_, traindescs

            self.alpha = (-np.log(0.01) / np.mean(dots[0, :] - dots[1, :])).item()
            self.centroids = nn.Parameter(torch.from_numpy(clsts_))
            self.conv.weight = nn.Parameter(torch.from_numpy(self.alpha * clsts_).unsqueeze(2).unsqueeze(3))
            self.conv.bias = None
        else:
            knn = NearestNeighbors(n_jobs=-1)
            clsts_ = clsts / np.linalg.norm(clsts, axis=1, keepdims=True)
            traindescs_ = traindescs / np.linalg.norm(traindescs, axis=1, keepdims=True)
            knn.fit(traindescs_)
            del traindescs_, traindescs
            dsSq = np.square(knn.kneighbors(clsts_, 2)[1])
            del knn
            self.alpha = (-np.log(0.01) / np.mean(dsSq[:, 1] - dsSq[:, 0])).item()
            self.centroids = nn.Parameter(torch.from_numpy(clsts_))
            del clsts_, dsSq

            self.conv.weight = nn.Parameter(
                (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
            )
            self.conv.bias = nn.Parameter(
                - self.alpha * self.centroids.norm(dim=1)
            )

    def forward(self, x):
        N, C = x.shape[:2]
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim
        # soft-assignment
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)

        x_flatten = x.view(N, C, -1)
        # calculate residuals to each clusters
        residual = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - \
                   self.centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
        # temp = torch.argmin(torch.norm(residual, p=2, dim=2), dim=1)
        # temp_ = torch.argmax(soft_assign, dim=1)
        residual *= soft_assign.unsqueeze(2)
        temp = residual  # To remove
        residual = torch.reshape(residual, (residual.size(0), -1, residual.size(-1)))  # To remove(?)
        # flatten
        residual = F.normalize(residual, p=2, dim=1)  # To remove or not(?)
        residual = torch.reshape(residual, (temp.size(0), temp.size(1), temp.size(2), temp.size(3)))  # To remove(?)
        vlad = residual.sum(dim=-1)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(x.size(0), -1)  # flatten
        # vlad = self.fc(vlad)
        vlad = self.bn(vlad)
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize
        return vlad


class HashSetNet(nn.Module):
    def __init__(self, base_model_architecture="resnet50_128", num_clusters=8, vlad_dim=128, vlad_v2=False, step_size=200):
        super(HashSetNet, self).__init__()

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
        self.net_vlad = NetVLAD(num_clusters=num_clusters, dim=dim, vlad_dim=vlad_dim,
                                vlad_v2=vlad_v2, normalize_input=True)

        self.iter_num = 0
        self.step_size = step_size
        self.gamma = 0.005
        self.power = 0.5
        self.init_scale = 1.0
        self.activation = nn.Tanh()
        self.scale = self.init_scale
        self.first_flag = True

    def forward(self, x, m):
        if self.training:
            if self.first_flag:
                self.iter_num += 1
            self.first_flag = not self.first_flag
        x, x_pre_flatten = self.base_model(x)
        x = x.view(int(x.shape[0] / m), m, x.shape[1]).unsqueeze(-1)
        x = x.permute(0, 2, 1, 3)
        # x = F.normalize(x, p=2, dim=1).squeeze()  # L2 normalize
        v_set = self.net_vlad(x)
        if self.iter_num % self.step_size == 0:
            self.scale = self.init_scale * (math.pow((1. + self.gamma * self.iter_num), self.power))
        v_set = self.activation(self.scale * v_set)
        return v_set


def acc_authentication(model, logisticReg, H0_id, H0_data, target, n_classes, v_set, v_f, m_set, n_batch_verif):
    H0_id = torch.stack(H0_id).flatten(start_dim=0, end_dim=1)
    H0_data = torch.stack(H0_data).flatten(start_dim=0, end_dim=1)
    indices = np.where(np.in1d(H0_id, target))[0]
    temp = np.arange(len(H0_id))
    temp[indices] = -1
    temp = temp[temp >= 0]
    classes_ = np.random.choice(len(temp), n_classes, replace=False)
    # H0_id = H0_id[temp[classes_]]
    H0_data = H0_data[temp[classes_]]

    temp = n_classes // n_batch_verif
    v_f0 = []
    model.eval()
    logisticReg.eval()
    with torch.no_grad():
        for i in range(n_batch_verif):
            v_f0.append(model(H0_data[i*temp:(i+1)*temp].to(device), m=1))  # single vector per query
    v_f0 = torch.sign(torch.stack(v_f0).flatten(start_dim=0, end_dim=1) + 1e-16)
    H0_claimed_group_id = np.random.randint(0, n_classes // m_set, size=n_classes).astype(np.int)
    # D00_ = torch.mm(v_set[H0_claimed_group_id], v_f0.t())
    # D00 = torch.diag(D00_).cpu()
    D00_ = logisticReg((torch.mm(v_set[H0_claimed_group_id], v_f0.t()) / v_f0.size(1)).unsqueeze(-1)).squeeze()
    D00 = torch.diag(F.sigmoid(D00_)).cpu()

    H1_group_id = np.repeat(np.arange(n_classes // m_set), m_set)
    # D11_ = torch.mm(v_set[H1_group_id], v_f.t())
    # D11 = torch.diag(D11_).cpu()
    D11_ = logisticReg((torch.mm(v_set[H1_group_id], v_f.t()) / v_f.size(1)).unsqueeze(-1)).squeeze()
    D11 = torch.diag(F.sigmoid(D11_)).cpu()
    D0 = np.sort(D00)[::-1]
    D1 = np.sort(D11)[::-1]

    Pfp = 0.01
    tau = D0[int(Pfp * n_classes)]
    Ptp01 = np.count_nonzero(D1 > tau) / n_classes
    Pfp = 0.05
    tau = D0[int(Pfp * n_classes)]
    Ptp05 = np.count_nonzero(D1 > tau) / n_classes
    return Ptp01, Ptp05


class LogisticReg(nn.Module):
    def __init__(self):
        super(LogisticReg, self).__init__()
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        x = self.fc(x)
        return x


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


def get_clusters(dataset, num_clusters, model_type="resnet50_128", batch_size=64, n_batches=500):
    initcache = os.path.join(ROOT_DIR, 'centroids',
                             model_type + '_' + '_' + str(num_clusters) + '_desc_cen.hdf5')
    model = Net(model_type).to(device)
    batch_sampler = BalanceBatchSampler(dataset=dataset, n_classes=64, n_samples=1,
                                        n_batches_epoch=n_batches)
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_sampler=batch_sampler, num_workers=2)
    nDescriptors = batch_size * n_batches
    if not os.path.exists(os.path.join(ROOT_DIR, 'centroids')):
        os.makedirs(os.path.join(ROOT_DIR, 'centroids'))
    with h5py.File(initcache, mode='w') as h5:
        with torch.no_grad():
            model.eval()
            print('====> Extracting Descriptors')
            dbFeat = h5.create_dataset("descriptors",
                                       [nDescriptors, model.encoder_dim],
                                       dtype=np.float32)

            for iteration, (data, target, img_file, class_id) in enumerate(data_loader):
                data = data.to(device)
                idx = iteration * batch_size
                dbFeat[idx:idx + batch_size, :] = F.normalize(model(data), p=2, dim=1).cpu().numpy()

        print('====> Clustering..')
        niter = 100
        kmeans = faiss.Kmeans(model.encoder_dim, num_clusters, niter=niter, verbose=False)
        kmeans.train(dbFeat[...])

        print('====> Storing centroids', kmeans.centroids.shape)
        h5.create_dataset('centroids', data=kmeans.centroids)
        print('====> Done!')
