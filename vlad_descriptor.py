import numpy as np
from sklearn.cluster import KMeans,MiniBatchKMeans
from sklearn import mixture
from metric import distEuclidean
import cv2 as cv
from sklearn.decomposition import PCA


def VLADDictionary(dataLearn, nClus, method='kMeans', distfun=distEuclidean):
    """
    @Parameters:
    dataLearn: N*d ndarray,each row is a sample.
    nClus: number of viusual words in the visual word dictinary.
    method: string, 'kMeans' or 'GMM',clustering algorithm used to create visual dictionary
    encode: hard or soft, used to assign each sample to cluster centers in hard or soft way.
    distfun: metric used to compute distance.(distEuclidean or distCosine)
    @Return:
        (k,dim) ndarray, cluster centers treated as codebook in viusual dictionary
    """
    method = str.lower(method)
    if method not in ('kmeans', 'gmm'):
        raise ValueError('Invalid clustering method for constructing visual dictionary')

    centers = None
    if method == 'kmeans':
        nSmp = dataLearn.shape[0]
        if nSmp < 3e4:
            km = KMeans(n_clusters=nClus, init='k-means++', n_init=3, n_jobs=-1)  # use all the cpus
        else:
            km = MiniBatchKMeans(n_clusters=nClus, init='k-means++', n_init=3)
        km.fit(dataLearn)
        centers = km.cluster_centers_
    else:
        gmm = mixture.GMM(n_components=nClus)
        gmm.fit(dataLearn)
        centers = gmm.means_
    return centers


def VLADEncoding(dataEncode, centers, encode='hard', distfun=distEuclidean, normalize=0):
    """
    @Parameters:
    dataEncode:N*d ndarray,each row is a sample,which is the data to be encoded.
    centers: nClus*nDim ndarray, clustering centers as the coodbook in the visual dictionary
    encode: hard or soft, used to assign each sample to cluster centers in hard or soft way.
    distfun: metric used to compute distance.(distEuclidean or distCosine)
    normalize: the normalization method
        0-Component-wise mass normalization.
        1-Square-rooting.
        2-Component-wise l2 normalization.
        3-Global l2 normalization.
    @Return:
        (k*dim,) ndarray, feature encoded in VLAD
    """
    nClus = centers.shape[0]
    if encode not in('hard', 'soft'):
        raise ValueError('Invalid value for VQ(hard or soft)')
    if dataEncode.ndim == 1:
        dataEncode = dataEncode[:, np.newaxis]
    nSmp, nDim = dataEncode.shape
    vlad = np.zeros((nClus, nDim))  # VLAD descriptors

    if encode == 'hard':
        # Vector quantization with hard or soft Assignment
        vq = np.zeros(nSmp)
        for idx in range(nSmp):
            mindist = np.Inf
            nn = -1
            dist_iter = map(lambda x: distfun(dataEncode[idx], x), centers)
            for (cnt, dist) in enumerate(dist_iter):
                if dist < mindist:
                    mindist = dist
                    nn = cnt
            vq[idx] = nn
        # Accumulate the residuals between descriptors and cluster centers
        for i in range(nClus):
            idx = vq == i
            data_diff = dataEncode[idx]-centers[i]
            vlad[i] = np.sum(data_diff, axis=0)
    else:  # VQ='soft'
        # Vector quantization with hard or soft Assignment
        vq = np.zeros((nSmp, nClus))
        for idx in range(nSmp):
            vq[idx] = np.array(list(map(lambda x: np.exp(-distfun(dataEncode[idx], x)), centers)))
            vq[idx] /= np.sum(vq[idx])
        # Accumulate the residuals between descriptors and cluster centers
        for k in range(nClus):
            diff_data = dataEncode-centers[k]
            for i in range(nSmp):
                diff_data[i] *= vq[i, k]
            vlad[k] = np.sum(diff_data, axis=0)
    # Normalize the finish the final encoding procedure
    if normalize == 0:
        # Each vector vk is divided by the total mass of features associated to it \sum_i q_{ik}
        for i in range(nClus):
            totalmass = sum(vq == i)
            vlad[i] /= totalmass
    elif normalize == 1:  # Apply sign(z)sqrt(|z|)is applied to all scalar components
        vlad = np.sign(vlad)*np.sqrt(np.abs(vlad))
    elif normalize == 2:  # Vectors vk are divided by their norm ∥v_k∥_2.
        for i in range(nClus):
            vlad[i] = vlad[i]/np.sqrt(np.sum(vlad[i]**2))
    elif normalize == 3:  # Component-wise l2 normalization.
        vlad /= np.sqrt(np.sum(vlad**2))
    else:
        raise ValueError('Invalid normalization option.')

    return vlad.flatten()


def apply_PCA(vlad_vector_data, components=128):
    vlad_d = vlad_vector_data
    pca = PCA(n_components=components)
    pca.fit(vlad_d)
    x1 = pca.transform(vlad_d)
    return x1


def extract_features(image_path):
    image = cv.imread(image_path, cv2.IMREAD_GRAYSCALE)
    des = image
    return des


if __name__ == '__main__':
    """
    @Parameters:
    dataLearn: N*d ndarray,each row is a sample.
    dataEncode:P*d ndarray,each row is a sample,which is the data to be encoded.
    nClus: number of viusual words in the visual word dictinary.
    method: string, 'kMeans' or 'GMM',clustering algorithm used to create visual dictionary
    encode: hard or soft, used to assign each sample to cluster centers in hard or soft way.
    distfun: metric used to compute distance.(distEuclidean or distCosine)
    normalize: the normalization method
        0-Component-wise mass normalization.
        1-Square-rooting.
        2-Component-wise l2 normalization.
        3-Global l2 normalization.
    @Return:
        (k*d,) ndarray, feature encoded in VLAD
    """
    dataLearn = np.random.rand(1000, 10)
    dataEncode = np.random.rand(1, 10)
    nClus = 5
    # Learn visual word dictionary with k-Means or GMM
    print("Clustering for visual word dictionary...")
    centers = VLADDictionary(dataLearn, nClus, method='kmeans', distfun=distEuclidean)
    
    print('Generating VLAD features...')
    vlad = VLADEncoding(dataEncode, centers, encode='hard', normalize=3)
    a = 2
