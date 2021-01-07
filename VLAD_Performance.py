from vlad_descriptor import *
import numpy as np
import pickle
from utils import vlad_group_representation, partition_data, hashing
from scipy.stats import zscore

if __name__ == '__main__':
    with open('dataset_LFW_VGG3.pkl', 'rb') as handle:
        dataset = pickle.load(handle)
    np.random.seed(10)
    n_Clus = 4
    dim = len(dataset['data_x'][0]) * n_Clus
    l = int(dim)
    Ptp01, Ptp05 = [], []
    S_x = int(l*0.7)  # 610
    group_member = 40  # 2,5,10,16,20,40
    W = np.random.random([dim, l])
    U, S, V = np.linalg.svd(W)
    W = U[:, :l]
    param = {'encode': 'soft', 'normalize': 3, 'n_Clus': n_Clus}
    # Assign data to groups
    data = np.array(dataset['data_x'])
    data = zscore(np.array(dataset['data_x']), axis=0)  # N*d
    groups = partition_data(data, group_member, partitioning='random')
    # Compute group representations
    group_vec, VLAD_Codebook = vlad_group_representation(data, groups, param)
    group_vec = np.array(group_vec).T
    group_vec = hashing(group_vec, W, S_x)
    # The embedding for H0 queries
    n_q0 = len(dataset['H0_id'])
    H0_data = zscore(np.array(dataset['H0_x']).T, axis=1)  # LFW
    # H0_data = zscore(np.array(dataset['H0_x']), axis=0)  # CFP
    H0_data = [VLADEncoding(np.expand_dims(H0_data[i, :], axis=0), VLAD_Codebook,
                            encode=param['encode'], normalize=param['normalize']) for i in range(n_q0)]
    Q0 = np.array(H0_data).T
    Q0 = hashing(np.array(H0_data).T, W, S_x)
    H0_claimed_id = np.random.randint(0, len(groups['ind']), size=n_q0).astype(np.int)
    D00 = np.linalg.norm(Q0 - group_vec[:, H0_claimed_id], axis=0)
    # The embedding for H1 queries
    n_q1 = len(dataset['H1_id'])
    H1_group_id = np.zeros(n_q1)
    H1_data = zscore(np.array(dataset['H1_x']), axis=0)
    H1_data = [VLADEncoding(np.expand_dims(H1_data[i, :], axis=0), VLAD_Codebook,
                            encode=param['encode'], normalize=param['normalize']) for i in range(n_q1)]
    Q1 = np.array(H1_data).T
    Q1 = hashing(np.array(H1_data).T, W, S_x)
    # Determine the group identity of H1 queries
    for i in range(len(groups['ind'])):
        group_id = [dataset['data_id'][x] for x in groups['ind'][i]]
        a = [n for n, x in enumerate(dataset['H1_id']) for y in group_id if x == y]
        for x in a:
            H1_group_id[x] = i
    D11 = np.linalg.norm(Q1 - group_vec[:, H1_group_id.astype(np.int)], axis=0)

    D0 = np.sort(D00)
    D1 = np.sort(D11)

    Pfp = 0.01
    tau = D0[int(Pfp * n_q0)]
    Ptp01 = np.count_nonzero(D1 <= tau) / n_q1
    Pfp = 0.05
    tau = D0[int(Pfp * n_q0)]
    Ptp05 = np.count_nonzero(D1 <= tau) / n_q1
    print('Ptp01:', Ptp01, 'Ptp05', Ptp05)
    a = 2
