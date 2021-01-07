import collections
import os
import numpy as np
import PIL.Image
import scipy.io
import torch
from torch.utils import data
import torchvision.transforms


class VGG_Faces2(data.Dataset):
    mean_bgr = np.array([91.4953, 103.8827, 131.0912])  # from resnet50_ft.prototxt
    mean_rgb = np.array([131.0912, 103.8827, 91.4953])  # from resnet50_ft.prototxt

    def __init__(self, root, split='train', transform=True,
                 horizontal_flip=False, upper=None):
        """
        :param root: dataset directory
        :param split: train or valid
        :param transform:
        :param horizontal_flip:
        :param upper: max number of image used for debug
        """
        assert os.path.exists(root), "root: {} not found.".format(root)
        self.root = root
        self.imgs_list = []
        self.sub_directory_list = os.listdir(self.root)
        self.id_label_dict = {}
        self.img_info = []
        self.split = split
        self._transform = transform
        self.horizontal_flip = horizontal_flip
        self.labels = []

        self.img_info = []
        # initialize id_label_dict and imgs_list
        for i, sub_directory in enumerate(self.sub_directory_list):
            imgs_list = os.listdir(os.path.join(self.root, sub_directory))
            self.id_label_dict[sub_directory.split("/")[-1]] = i
            for img in imgs_list:
                img_file = os.path.join(self.root, sub_directory, img)  # e.g. train/n004332/0317_01.jpg
                class_id = sub_directory  # like n004332
                label = self.id_label_dict[class_id]
                self.labels.append(label)

                self.img_info.append({
                    'cid': class_id,
                    'img': img_file,
                    'lbl': label,
                })
            if i % 100 == 0:
                print("processing: {} identities for {}".format(i, self.split))
            if upper and i == upper - 1:  # for debug purpose
                break

    def __len__(self):
        return len(self.img_info)

    def __getitem__(self, index):
        info = self.img_info[index]
        img_file = info['img']
        img = PIL.Image.open(os.path.join(self.root, img_file))
        img = torchvision.transforms.Resize(256)(img)
        if self.split == 'train':
            img = torchvision.transforms.CenterCrop(224)(img)
            # img = torchvision.transforms.RandomGrayscale(p=0.2)(img)
        else:
            img = torchvision.transforms.CenterCrop(224)(img)
        if self.horizontal_flip:
            img = torchvision.transforms.functional.hflip(img)

        img = np.array(img, dtype=np.uint8)
        assert len(img.shape) == 3  # assumes color images and no alpha channel

        label = info['lbl']
        class_id = info['cid']
        if self._transform:
            return self.transform(img), label, img_file, class_id
        else:
            return img, label, img_file, class_id

    def transform(self, img):
        # img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float32)
        img -= self.mean_rgb
        img = img.transpose(2, 0, 1)  # C x H x W
        img = torch.from_numpy(img).float()
        return img

    def untransform(self, img, lbl):
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        img += self.mean_rgb
        img = img.astype(np.uint8)
        # img = img[:, :, ::-1]  # mean_bgr
        return img, lbl
