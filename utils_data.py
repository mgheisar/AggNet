import numpy as np
from torch.utils.data.sampler import BatchSampler
import torch
import os


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BalanceBatchSampler(BatchSampler):
    """
    batch sampler, randomly select n_classes, and n_samples each class
    """

    def __init__(self, dataset, n_classes, n_samples, n_batches_epoch=None):
        try:
            self.labels = dataset.labels  # labels
        except AttributeError:
            try:
                self.labels = dataset.targets  # labels
            except AttributeError:
                self.labels = [img[1]for img in dataset.imgs]
        self.labels_array = np.array(list(self.labels))
        self.labels_set = list(set(self.labels_array))
        self.labels_set = [label for label in self.labels_set if len(np.where(self.labels_array == label)[0]) >= n_samples]
        self.labels_to_indices = {label: np.where(self.labels_array == label)[0]
                                  for label in self.labels_set if len(np.where(self.labels_array == label)
                                                                      [0]) >= n_samples}
        for l in self.labels_set:
            np.random.shuffle(self.labels_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.batch_size = n_classes * n_samples
        self.dataset = dataset
        self.new_dataset_size = np.array([len(v) for k, v in self.labels_to_indices.items()]).sum()
        self.n_batches_epoch = n_batches_epoch
        if self.n_batches_epoch is None:
            self.n_batches_epoch = self.new_dataset_size // self.batch_size  # len(self.dataset)
        print('n_batches_epoch', self.n_batches_epoch)
        self.batch_indices = []
        self.unselected_classes = self.labels_set.copy()
        for i in range(self.n_batches_epoch):
            if len(self.unselected_classes) < self.n_classes:
                # Making sure that one epoch will used all the classes
                np.random.seed(0)  # (2)When number of train identities is dividable to the n_classes
                # To assign only particular identities to same group(to not change groups in different batches in the same epoch)
                self.classes1 = self.unselected_classes.copy()
                self.unselected_classes1 = self.labels_set.copy()
                [self.unselected_classes1.remove(element) for element in self.classes1]
                self.n_classes2 = self.n_classes - len(self.classes1)
                self.classes2 = np.random.choice(self.unselected_classes1, self.n_classes2, replace=False)
                self.unselected_classes = self.labels_set.copy()
                [self.unselected_classes.remove(element) for element in self.classes2]
            else:
                self.classes = np.random.choice(self.unselected_classes, self.n_classes, replace=False)
                [self.unselected_classes.remove(element) for element in self.classes]
            indices = []
            for class_ in self.classes:
                indices.extend(self.labels_to_indices[class_][self.used_label_indices_count[class_]:
                                                              self.used_label_indices_count[class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.labels_to_indices[class_]):
                    np.random.shuffle(self.labels_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            self.batch_indices.append(indices)

    def __iter__(self):
        self.count = 0
        for i in range(self.n_batches_epoch):
            yield self.batch_indices[i]
            self.count += self.batch_size

    def __len__(self):
        return self.n_batches_epoch


class Reporter(object):
    def __init__(self, ckpt_root, exp, monitor):
        self.ckpt_root = ckpt_root
        self.exp_path = os.path.join(self.ckpt_root, exp)
        self.run_list = os.listdir(self.exp_path)
        self.selected_ckpt = None
        self.selected_epoch = None
        self.selected_log = None
        self.selected_run = None
        self.last_epoch = 0
        self.last_loss = 0
        self.monitor = monitor

    def select_best(self, run=""):

        """
        set self.selected_run, self.selected_ckpt, self.selected_epoch
        :param run:
        :return:
        """

        matched = []
        for fname in self.run_list:
            if fname.startswith(run) and fname.endswith('tar'):
                matched.append(fname)

        loss = []
        import re
        for s in matched:
            if re.search('-1', s):
                matched.remove(s)
            else:
                if self.monitor == 'loss':
                    acc_str = re.search('loss_(.*)\.tar', s).group(1)
                elif self.monitor == 'acc':
                    acc_str = re.search('acc_(.*)\.tar', s).group(1)
                loss.append(float(acc_str))

        loss = np.array(loss)
        if self.monitor == 'loss':
            best_idx = np.argmin(loss)
            best_fname = matched[best_idx]
            self.selected_run = best_fname.split(',')[0]
            self.selected_epoch = int(re.search('Epoch_(.*),loss', best_fname).group(1))
        elif self.monitor == 'acc':
            best_idx = np.argmax(loss)
            best_fname = matched[best_idx]
            self.selected_run = best_fname.split(',')[0]
            self.selected_epoch = int(re.search('Epoch_(.*),acc', best_fname).group(1))

        ckpt_file = os.path.join(self.exp_path, best_fname)
        self.selected_ckpt = ckpt_file
        return self

    def select_last(self, run=""):

        """
        set self.selected_run, self.selected_ckpt, self.selected_epoch
        :param run:
        :return:
        """
        matched = []
        for fname in self.run_list:
            if fname.startswith(run+',') and fname.endswith('tar'):
                matched.append(fname)

        import re
        for s in matched:
            if re.search('last_Epoch', s):
                if self.monitor == 'loss':
                    epoch = re.search('last_Epoch_(.*),loss', s).group(1)
                    loss = re.search('loss_(.*)', s).group(1)
                elif self.monitor == 'acc':
                    epoch = re.search('last_Epoch_(.*),acc', s).group(1)
                    loss = re.search('acc_(.*)', s).group(1)
                last_fname = s

        self.selected_run = last_fname.split(',')[0]
        self.last_epoch = epoch
        self.last_loss = loss

        ckpt_file = os.path.join(self.exp_path, last_fname)
        self.selected_ckpt = ckpt_file

        return self
