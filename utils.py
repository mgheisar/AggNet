from vlad_descriptor import *
import numpy as np
import torch
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def vlad_group_representation(data, group, params):
    data_Learn = np.array(data)
    # Learn visual word dictionary with k-Means or GMM
    Centers = VLADDictionary(data_Learn, nClus=params['n_Clus'], method='kmeans', distfun=distEuclidean)
    group_rep = []
    for i in range(len(group['data'])):
        group_rep.append(VLADEncoding(np.array(group['data'][i]), Centers, encode=params['encode'],
                                      normalize=params['normalize']))
    return group_rep, Centers


def baseline_group_representation(group, params):
    data = np.array(group['data']).T
    if params['method'] == 'EoA':
        group_rep = aggregation_vec(data, aggregation_mode=params['agg'])
        group_rep = hashing(group_rep, params['W'], params['S_x'])
    elif params['method'] == 'AoE':
        Hash = hashing(data, params['W'], params['S_x'])
        group_rep = aggregation_emb(Hash, aggregation_mode=params['agg'])
    return group_rep


def partition_data(data, m, partitioning='random'):
    group = {'data': [], 'ind': []}
    M = int(len(data) / m)
    if partitioning == 'random':
        a = np.random.permutation(len(data))
    for i in range(M):
        group['data'].append([data[x] for x in a[i * m: (i + 1) * m]])
        group['ind'].append([x for x in a[i * m: (i + 1) * m]])
    if len(data) % m is not 0:
        group['data'].append([data[x] for x in a[(i + 1) * m:]])
        group['ind'].append([x for x in a[(i + 1) * m:]])
    return group


def hashing(X, W, S_x):
    # W : d*l
    # X : d*n
    shape = W.shape[1:] + X.shape[1:]
    Hash = np.matmul(W.T, X.reshape(X.shape[0], -1)).reshape(shape)
    l = Hash.shape[0]
    flag_D2 =False
    # set the l-S_x smallest values to zero
    if Hash.ndim == 2:
        flag_D2 = True
        Hash = np.expand_dims(Hash, axis=2)
    for i in range(Hash.shape[2]):
        A = Hash[:, :, i]
        for column in A.T: column[np.argsort(abs(column))[:l-S_x]] = 0
        Hash[:, :, i] = A
    Hash = np.sign(Hash)
    if flag_D2:
        Hash = np.squeeze(Hash)
    return Hash


def aggregation_vec(X, aggregation_mode='sum'):
    # X : d*m*M
    if aggregation_mode == 'sum':
        Aggregated_vec = np.sum(X, axis=1)

    elif aggregation_mode == 'pseudo':
        Aggregated_vec = np.zeros([X.shape[0], X.shape[2]])
        for i in range(X.shape[2]):
            Aggregated_vec[:, i] = np.matmul(np.linalg.pinv(X[:, :, i]).T, np.ones(X.shape[1]))
    return Aggregated_vec


def aggregation_emb(Hash, aggregation_mode='sum'):
    if aggregation_mode == 'sum':
        Aggregated_vec = np.sign(np.sum(Hash, axis=1))

    elif aggregation_mode == 'majority':
        Aggregated_vec = np.zeros([Hash.shape[0], Hash.shape[2]])
        for i in range(Hash.shape[0]):
            for j in range(Hash.shape[2]):
                vals, counts = np.unique(Hash[i, :, j], return_counts=True)
                Aggregated_vec[i, j] = vals[np.argmax(counts)]

    return Aggregated_vec
