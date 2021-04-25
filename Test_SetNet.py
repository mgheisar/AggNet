import os
import torchvision
import torch
from AggNet import SetNet, get_clusters, LogisticReg, acc_authentication
from utils_data import BalanceBatchSampler, Reporter
import numpy as np
import yaml
import argparse
from vgg_face2 import VGG_Faces2
from itertools import chain
import h5py
import multiprocessing

torch.manual_seed(0)
if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force=True)
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    # print(ROOT_DIR)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    core_number = multiprocessing.cpu_count()
    print('core number:', core_number)
    #  --------------------------------------------------------------------------------------
    # Arguments
    #  --------------------------------------------------------------------------------------
    with open(r'{}/args.yaml'.format(ROOT_DIR)) as file:
        args_list = yaml.load(file, Loader=yaml.FullLoader)
    dataroot = args_list['dataroot']
    model_type = args_list['model_type']
    n_epoch = args_list['n_epoch']
    n_classes = args_list['n_classes']
    n_samples = args_list['n_samples']
    m_set = args_list['m_set']
    run_name = args_list['run_name']
    exp_name = args_list['exp_name']
    n_batches_valid = args_list['n_batches_valid']
    upper_vgg = args_list['upper_vgg']
    num_clusters = args_list['num_clusters']
    clustering = args_list['clustering']
    vlad_v2 = args_list['vlad_v2']
    lossFun = args_list['loss']
    num_workers = args_list['num_workers']
    vlad_dim = args_list['vlad_dim']

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', '--model', type=str, default=model_type,
                        help='model name (default: "resnet50_128")')
    parser.add_argument('--run_name', '--rn', type=str, default=run_name,
                        help='The name for this run (default: "Run01")')
    parser.add_argument('--m_set', '--m', type=int, default=m_set,
                        help='the group size')
    parser.add_argument('--n_batches_valid', '--n_b_valid', type=int, default=n_batches_valid,
                        help='Number of batches per epoch for validation')
    parser.add_argument('--upper_vgg', '--u_vgg', type=int, default=upper_vgg,
                        help='Number of images loaded from VGG-Face2')
    parser.add_argument('--n_batch_verif', '--n_batch_verif', type=int,
                        help='Number of batches for verification (default: 8)')
    parser.add_argument('--loss', '--loss', type=str, default=lossFun,
                        help='loss function (default: "loss_bc")')
    parser.add_argument('--pooling', '--pooling', type=str, default='vlad',
                        help='pooling method (default: "vlad")')
    parser.add_argument('--n_classes', '--n_classes', type=int, default=n_classes,
                        help='Number of classes in batch (default: 32)')

    args = parser.parse_args()
    model_type = args.model_type
    run_name = args.run_name
    m_set = args.m_set
    n_batches_valid = args.n_batches_valid
    upper_vgg = args.upper_vgg
    n_batch_verif = args.n_batch_verif
    lossFun = args.loss
    pooling = args.pooling
    n_classes = args.n_classes

    if lossFun == 'loss_bc':
        from loss import loss_bc as loss_fn
    elif lossFun == 'loss_bc_fb':
        from loss import loss_bc_fb as loss_fn
    elif lossFun == 'loss_auc_max_v1':
        from loss import loss_auc_max_v1 as loss_fn
    elif lossFun == 'loss_AUCPRHingeLoss':
        from loss import loss_AUCPRHingeLoss as loss_fn

    if n_batches_valid == 0:
        n_batches_valid = None
    if upper_vgg == 0:
        upper_vgg = None
    #  --------------------------------------------------------------------------------------
    # Load train dataset
    #  --------------------------------------------------------------------------------------
    # VGG Face2
    training_dataset_root = '/nfs/nas4/marzieh/marzieh/VGG_Face2/train/'
    dataset_train = VGG_Faces2(training_dataset_root, split='train', upper=upper_vgg)
    exp_name = 'lfw'
    if exp_name == 'lfw':
        mean_rgb = (0.485, 0.456, 0.406)  # (0.5, 0.5, 0.5)
        std_rgb = (0.229, 0.224, 0.225)  # (0.5, 0.5, 0.5)
        dataset_validation = torchvision.datasets.ImageFolder(root=dataroot,
                                                              transform=torchvision.transforms.Compose([
                                                                  torchvision.transforms.Resize(256),
                                                                  torchvision.transforms.CenterCrop(224),
                                                                  torchvision.transforms.ToTensor(),
                                                                  torchvision.transforms.Normalize(mean=mean_rgb,
                                                                                                   std=std_rgb)]
                                                              ))
    elif exp_name == 'vgg2':
        validation_dataset_root = '/nfs/nas4/marzieh/marzieh/VGG_Face2/test/'
        dataset_validation = VGG_Faces2(validation_dataset_root, split='validation', upper=upper_vgg)
    #  --------------------------------------------------------------------------------------
    # Batch Sampling: n_samples * n_samples
    #  --------------------------------------------------------------------------------------
    batch_size = n_classes * n_samples
    batch_sampler_v = BalanceBatchSampler(dataset=dataset_validation, n_classes=n_classes, n_samples=n_samples,
                                        n_batches_epoch=n_batches_valid)
    validation_loader = torch.utils.data.DataLoader(dataset_validation, batch_sampler=batch_sampler_v,
                                                    num_workers=num_workers)
    batch_sampler_H0t = BalanceBatchSampler(dataset=dataset_train, n_classes=n_classes * 2, n_samples=1,
                                        n_batches_epoch=n_batch_verif)
    H0_loader_train = torch.utils.data.DataLoader(dataset_train, batch_sampler=batch_sampler_H0t,
                                            num_workers=num_workers)
    # batch_sampler_H0v = BalanceBatchSampler(dataset=dataset_validation, n_classes=n_classes * 2, n_samples=1,
    #                                     n_batches_epoch=n_batch_verif)
    # H0_loader_validation = torch.utils.data.DataLoader(dataset_validation, batch_sampler=batch_sampler_H0v,
    #                                                    num_workers=num_workers)
    batch_sampler_H0v = BalanceBatchSampler(dataset=dataset_train, n_classes=n_classes * 2, n_samples=1,
                                            n_batches_epoch=n_batch_verif)
    H0_loader_validation = torch.utils.data.DataLoader(dataset_train, batch_sampler=batch_sampler_H0v,
                                                  num_workers=num_workers)
    H0_id_v, H0_data_v = [], []
    dataloader_H0_t = iter(H0_loader_train)
    dataloader_H0_v = iter(H0_loader_validation)
    for i in range(n_batch_verif):
        data = next(dataloader_H0_v)
        H0_id_v.append(data[1])
        H0_data_v.append(data[0])
    #  --------------------------------------------------------------------------------------
    # Model Definitions
    #  --------------------------------------------------------------------------------------
    model = SetNet(base_model_architecture=model_type, num_clusters=num_clusters, vset_dim=vlad_dim,
                   vlad_v2=vlad_v2, pooling=pooling)
    logisticReg = LogisticReg()
    model.to(device)
    logisticReg.to(device)
    #  --------------------------------------------------------------------------------------
    #  Loading the model
    #  --------------------------------------------------------------------------------------
    reporter = Reporter(ckpt_root=os.path.join(ROOT_DIR, 'ckpt'),
                        exp='vgg2', monitor='acc')  # monitor='auc'
    best_model_filename = reporter.select_best(run=run_name).selected_ckpt
    model.load_state_dict(torch.load(best_model_filename)['model_state_dict'])
    reporter = Reporter(ckpt_root=os.path.join(ROOT_DIR, 'ckpt'),
                        exp='vgg2', monitor='acc')  # monitor='auc'
    best_model_filename = reporter.select_best(run=run_name + '_lr').selected_ckpt
    logisticReg.load_state_dict(torch.load(best_model_filename)['model_state_dict'])
    # ---------  Test-----------------
    # model.eval()
    # logisticReg.eval()
    tot_loss, tot_acc = 0, 0
    n_batches = len(validation_loader)
    Ptp01, Ptp05, AUC = np.zeros(n_batches // n_batch_verif), np.zeros(n_batches // n_batch_verif), np.zeros(n_batches // n_batch_verif)
    vs, vf, tg = [], [], []
    idx = -1
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(validation_loader):
            data_set = data[np.arange(0, batch_size, n_samples)].to(device)
            data_query = data[np.arange(1, batch_size, n_samples)].to(device)
            v_set = model(data_set, m=m_set)  # single vector per set
            v_f = model(data_query, m=1)  # single vector per query
            Sim = torch.mm(v_set, v_f.t())
            # output = Sim
            output = logisticReg(Sim.unsqueeze(-1)).squeeze()
            loss_outputs, accuracy = loss_fn(output, len(v_f), m_set)
            tot_acc += accuracy
            tot_loss += loss_outputs

            vs.append(v_set)
            vf.append(v_f)
            tg.append(target)
            if (batch_idx + 1) % n_batch_verif == 0:
                idx += 1
                vs = torch.stack(vs).flatten(start_dim=0, end_dim=1)
                vf = torch.stack(vf).flatten(start_dim=0, end_dim=1)
                tg = torch.stack(tg).flatten(start_dim=0, end_dim=1)

                Ptp01[idx], Ptp05[idx], AUC[idx] = acc_authentication(model, logisticReg, H0_id_v, H0_data_v,
                                                                  tg, vf.size(0), vs, vf, m_set, n_batch_verif)
                vs, vf, tg = [], [], []
    avg_loss = tot_loss / n_batches
    avg_acc = tot_acc / n_batches
    print('Evaluation--->avg_loss: %.3f' % avg_loss, 'avg_acc: %.3f' % avg_acc,
          ' ptp01: %.3f' % np.mean(Ptp01), 'ptp05: %.3f' % np.mean(Ptp05)
          , ' auc: %.3f' % np.mean(AUC))

# avg_loss: 0.033 avg_acc: 0.960  ptp01: 0.728 ptp05: 0.919  auc: 0.980
