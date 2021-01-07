import scipy.io as sio
import re
from FeatureExtraction import *
import pickle


np.random.seed(1)
tau1 = 0.9
tau2 = 1
emb_dim = 512
model_name = "vgg16"  # resnet50, senet50, vgg16
root_path = '/home/mgheisar/workspace/face_exp/cropped_LFW_images/'
imglist = sio.loadmat(root_path+'imglist.mat')['imglist']
# a list of all identities associated with the number of his/her images in the database
names = {}
lfw_names = []
with open(root_path+"lfw-names.txt", "r") as f:
    for line in f:
        (key, val) = line.split()
        names[key] = val
        # Choose identities with at least 2 images
        if int(val) > 2:
            lfw_names.append(key)

# the local path of each image
imglist = list(map(lambda x: x[0][0], imglist))
# find the identity of each image
img_identity = [re.search('lfw/lfw/(.*)/', img).group(1) for img in imglist]
# # Extract features
features = feature_extraction(root_path, imglist, emb_dim=emb_dim, model_name=model_name)
enrolled_ind = np.zeros(len(imglist))
dataset = {'data_id': [], 'data_x': [], 'data_ind': [], 'H1_id': [], 'H1_x': [],
           'H1_ind': [], 'H0_id': [], 'H0_x': [], 'H0_ind': []}
for i in range(len(lfw_names)):
    indiv_images = np.array([n for n, x in enumerate(img_identity) if lfw_names[i] == x])
    enrolled_ind[indiv_images] = 1
    # [print(img_identity[n]) for n in indiv_images]
    permuted = indiv_images[np.random.permutation(len(indiv_images))]
    ix = 0
    flag_h1 = False
    while ix < len(indiv_images) and not flag_h1:
        k = ix+1
        while ix+1 <= k < len(indiv_images) and not flag_h1:
            data_id = lfw_names[i]
            data_x = features[:, permuted[ix]]
            data_ind = permuted[ix]
            alpha = np.dot(features[:, permuted[ix]], features[:, permuted[k]])
            if tau1 < alpha <= tau2:
                dataset['H1_id'].append(lfw_names[i])
                dataset['H1_x'].append(features[:, permuted[k]])
                dataset['H1_ind'].append(permuted[k])
                flag_h1 = True
            k += 1
        ix += 1
    dataset['data_id'].append(data_id)
    dataset['data_x'].append(data_x)
    dataset['data_ind'].append(data_ind)

H0_ind = np.array(np.where(enrolled_ind == 0)).squeeze()
a = np.random.permutation(len(H0_ind))
H0_ind = H0_ind[a]
dataset['H0_ind'] = H0_ind[:len(dataset['H1_ind'])]
dataset['H0_id'] = [img_identity[i] for i in dataset['H0_ind']]
dataset['H0_x'] = features[:, dataset['H0_ind']]

f = open("dataset_LFW_VGG3.pkl", "wb")
pickle.dump(dataset, f)
f.close()





