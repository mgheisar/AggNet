import torch
import models
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable


def feature_extraction(root_path, imglist, emb_dim=1024, model_name="resnet50"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    to_tensor = transforms.ToTensor()
    scaler = transforms.Scale((224, 224))

    feat = np.zeros([emb_dim, len(imglist)])
    model = models.FeatureExtractor(embedding_dimension=emb_dim, model_architecture=model_name, pretrained=True)
    model.to(device)

    model.eval()
    with torch.no_grad():
        for i in range(len(imglist)):
            img = Image.open(root_path + imglist[i])
            x = Variable(to_tensor(scaler(img))).unsqueeze(0)
            # x = np.expand_dims(img, axis=0)
            f_vec = model(x)
            feat[:, i] = f_vec
    return feat
