from keras_vggface import VGGFace
from keras.preprocessing import image
from keras import backend as K
import numpy as np
from sklearn.decomposition import PCA


def feature_extraction(root_path, imglist, emb_dim=1024, model_name="vgg16"):
    K.common.image_dim_ordering = 'tf'
    # Features by keras vgg16 trained on VGGFACE2
    # vggface = VGGFace(model=model_name, include_top=False, input_shape=(224, 224, 3), pooling='avg')
    # get_output = K.function([vggface.layers[0].input], [vggface.layers[-1].output])

    vggface = VGGFace(model=model_name)
    get_output = K.function([vggface.layers[0].input], [vggface.layers[-4].output])

    feat = []
    for i in range(len(imglist)):
        img = image.load_img(root_path+imglist[i], target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x[:, :, :, ::-1]
        f_vec = get_output([x])[0]
        f_vec = np.squeeze(f_vec)
        norm = np.sqrt(f_vec.dot(f_vec))
        f_vec = f_vec/norm
        feat.append(f_vec)

    # features = np.array(feat)

    feat = np.array(feat)
    pca = PCA(n_components=emb_dim)
    pca.fit(feat)
    features = pca.transform(feat)

    features = features.T
    features = features / np.linalg.norm(features, axis=0)
    return features
