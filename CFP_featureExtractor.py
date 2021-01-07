from FeatureExtraction import *
import pickle

np.random.seed(1)
tau1 = 0.9
tau2 = 1
emb_dim = 1024
model_name = "resnet50"  # resnet50, senet50, vgg16
H0_number = 100
Individuals = 500
ImagePerIndividual = 10
root_path = '/home/mgheisar/workspace/face_exp/cfp-dataset/Data/Images/'
# imglist =
# a list of all identities associated with the number of his/her images in the database
cfp_names = ['{:03d}'.format(x) for x in np.arange(1, Individuals+1).tolist()]
dirlist = [x+'/frontal/' for x in cfp_names]
imgs = ['{:02d}'.format(x) for x in np.arange(1, ImagePerIndividual+1).tolist()]
imglist = [x+y+'.jpg' for x in dirlist for y in imgs]

features = feature_extraction(root_path, imglist, emb_dim=emb_dim, model_name=model_name)

dataset = {'data_id': [], 'data_x': [], 'data_ind': [], 'H1_id': [], 'H1_x': [],
           'H1_ind': [], 'H0_id': [], 'H0_x': [], 'H0_ind': []}

rnd = np.random.permutation(len(cfp_names))
dataset['H0_id'] = [cfp_names[i] for i in rnd[:H0_number]]
dataset['data_id'] = [cfp_names[i] for i in rnd[H0_number:]]
for i in range(H0_number):
    rnd = np.random.randint(0, ImagePerIndividual)
    indx = (int(dataset['H0_id'][i])-1) * 10 + rnd
    dataset['H0_ind'].append(indx)
    dataset['H0_x'].append(features[:, indx])

for i in range(len(dataset['data_id'])):
    indiv_images = (int(dataset['data_id'][i])-1) * 10 + np.arange(0, ImagePerIndividual)
    permuted = indiv_images[np.random.permutation(ImagePerIndividual)]
    ix = 0
    flag_h1 = False
    while ix < ImagePerIndividual and not flag_h1:
        k = ix+1
        while ix+1 <= k < ImagePerIndividual and not flag_h1:
            data_x = features[:, permuted[ix]]
            data_ind = permuted[ix]
            alpha = np.dot(features[:, permuted[ix]], features[:, permuted[k]])
            if tau1 < alpha <= tau2:
                dataset['H1_id'].append(dataset['data_id'][i])
                dataset['H1_x'].append(features[:, permuted[k]])
                dataset['H1_ind'].append(permuted[k])
                flag_h1 = True
            k += 1
        ix += 1
    dataset['data_x'].append(data_x)
    dataset['data_ind'].append(data_ind)

f = open("dataset_CFP.pkl", "wb")
pickle.dump(dataset, f)
f.close()
