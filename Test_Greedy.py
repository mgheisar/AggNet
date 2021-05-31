# Test LFW on HashSetNet(Greedy)
# --rn="Run010" --n_classes=32 --m=4 --n_b_valid=2000 --n_batch_verif=4
import os
import torchvision
import torch
from Greedy import HashSetNet, LogisticReg, acc_authentication
from utils_data import BalanceBatchSampler, Reporter
import numpy as np
import torch.nn.functional as F
import yaml
import argparse
from vgg_face2 import VGG_Faces2
import multiprocessing

if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)
    torch.multiprocessing.set_start_method('spawn', force=True)
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    # print(ROOT_DIR)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    core_number = multiprocessing.cpu_count()
    # print('core number:', core_number)
    #  --------------------------------------------------------------------------------------
    # Arguments
    #  --------------------------------------------------------------------------------------
    with open(r'{}/args.yaml'.format(ROOT_DIR)) as file:
        args_list = yaml.load(file, Loader=yaml.FullLoader)
    dataroot = args_list['dataroot']
    model_type = args_list['model_type']
    n_classes = args_list['n_classes']
    n_samples = args_list['n_samples']
    m_set = args_list['m_set']
    run_name = args_list['run_name']
    exp_name = args_list['exp_name']
    n_batches_valid = args_list['n_batches_valid']
    upper_vgg = args_list['upper_vgg']
    lossFun = args_list['loss']
    num_workers = args_list['num_workers']
    num_clusters = args_list['num_clusters']
    vlad_v2 = args_list['vlad_v2']
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
                        help='Number of batches for verification')
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
    elif lossFun == 'loss_auc_max_v1':
        from loss import loss_auc_max_v1 as loss_fn
    elif lossFun == 'loss_AUCPRHingeLoss':
        from loss import loss_AUCPRHingeLoss as loss_fn

    if n_batches_valid == 0:
        n_batches_valid = None
    if upper_vgg == 0:
        upper_vgg = None
    #  --------------------------------------------------------------------------------------
    # Load datasets
    #  --------------------------------------------------------------------------------------
    print('1')
    exp_name = 'celeba'  # 'vgg2'
    test_name = 'lfw'
    mean_rgb = (0.485, 0.456, 0.406)  # (0.5, 0.5, 0.5) (0.485, 0.456, 0.406)
    std_rgb = (0.229, 0.224, 0.225)  # (0.5, 0.5, 0.5) (0.229, 0.224, 0.225)
    preprocess = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=mean_rgb, std=std_rgb),
    ])
    if test_name == 'celeba':
        dataroot = '/nfs/nas4/marzieh/marzieh/celebA/'
        dataset_validation = torchvision.datasets.CelebA(root=dataroot, split='test',
                    target_type='identity', transform=preprocess)  # split='test' 'valid'
    elif test_name == 'lfw':
        dataroot = '/nfs/nas4/marzieh/marzieh/VGG_Face2/lfw/lfw-deepfunneled/'
        dataset_validation = torchvision.datasets.ImageFolder(root=dataroot, transform=preprocess)
    elif test_name == 'cfp':
        dataroot = '/nfs/nas4/marzieh/marzieh/cfp/cfpf-dataset/Data/Images/'
        dataset_validation = torchvision.datasets.ImageFolder(root=dataroot, transform=preprocess)
    elif test_name == 'casia':
        dataroot = '/nfs/nas4/marzieh/marzieh/CASIA/CASIA-WebFace/'
        dataset_validation = torchvision.datasets.ImageFolder(root=dataroot, transform=preprocess)
    elif test_name == 'vgg2':
        validation_dataset_root = '/nfs/nas4/marzieh/marzieh/VGG_Face2/test/'
        dataset_validation = VGG_Faces2(validation_dataset_root, split='validation', upper=upper_vgg)

    #  --------------------------------------------------------------------------------------
    # Batch Sampling: n_samples * n_samples
    #  --------------------------------------------------------------------------------------
    print('2')
    batch_size = n_classes * n_samples
    batch_sampler_v = BalanceBatchSampler(dataset=dataset_validation, n_classes=n_classes, n_samples=n_samples,
                                          n_batches_epoch=n_batches_valid)
    validation_loader = torch.utils.data.DataLoader(dataset_validation, batch_sampler=batch_sampler_v,
                                                    num_workers=num_workers)
    batch_sampler_H0v = BalanceBatchSampler(dataset=dataset_validation, n_classes=n_classes * 2, n_samples=1,
                                            n_batches_epoch=n_batch_verif)
    H0_loader_validation = torch.utils.data.DataLoader(dataset_validation, batch_sampler=batch_sampler_H0v,
                                                       num_workers=num_workers)
    H0_id_v, H0_data_v = [], []
    dataloader_H0_v = iter(H0_loader_validation)
    for i in range(n_batch_verif):
        data = next(dataloader_H0_v)
        H0_id_v.append(data[1])
        H0_data_v.append(data[0])
    #  --------------------------------------------------------------------------------------
    # Model Definitions
    #  --------------------------------------------------------------------------------------
    print('3')
    model = HashSetNet(base_model_architecture=model_type, num_clusters=num_clusters, vset_dim=vlad_dim,
                       vlad_v2=vlad_v2, pooling=pooling)
    logisticReg = LogisticReg()
    model.to(device)
    logisticReg.to(device)
    #  --------------------------------------------------------------------------------------
    #  Loading the model
    #  --------------------------------------------------------------------------------------
    # reporter.monitor = 'auc' or 'acc' ????????????
    reporter = Reporter(ckpt_root=os.path.join(ROOT_DIR, 'ckpt'),
                        exp=exp_name, monitor='acc')
    best_model_filename = reporter.select_best(run=run_name).selected_ckpt
    # print(best_model_filename)
    model.load_state_dict(torch.load(best_model_filename)['model_state_dict'])
    reporter = Reporter(ckpt_root=os.path.join(ROOT_DIR, 'ckpt'),
                        exp=exp_name, monitor='acc')
    best_model_filename = reporter.select_best(run=run_name + '_lr').selected_ckpt
    # print(best_model_filename)
    logisticReg.load_state_dict(torch.load(best_model_filename)['model_state_dict'])
    # model.eval()
    # logisticReg.eval()
    tot_loss, tot_acc = 0, 0
    n_batches = len(validation_loader)
    Ptp01, Ptp05, Ptp1, AUC = np.zeros(n_batches // n_batch_verif), np.zeros(n_batches // n_batch_verif), np.zeros(
        n_batches // n_batch_verif), np.zeros(n_batches // n_batch_verif)
    vs, vf, tg = [], [], []
    idx = -1
    print('4')
    with torch.no_grad():
        for batch_idx, value in enumerate(validation_loader):
            print('5')
            data = value[0]
            target = value[1]
            data_set = data[np.arange(0, batch_size, n_samples)].to(device)
            data_query = data[np.arange(1, batch_size, n_samples)].to(device)
            v_set, code_set = model(data_set, m=m_set)  # single vector per set
            v_f, code_f = model(data_query, m=1)  # single vector per query
            Sim = torch.mm(F.normalize(code_set, p=2, dim=1), F.normalize(code_f, p=2, dim=1).t())
            output = logisticReg(Sim.unsqueeze(-1)).squeeze()
            loss1, accuracy = loss_fn(output, len(code_f), m_set)

            # h = torch.cat([v_set, v_f], dim=0)
            # loss2 = torch.mean(torch.abs(torch.pow(torch.abs(h) - Variable(torch.ones(h.size()).cuda()), 3)))
            # loss_outputs = loss1 + alpha * loss2
            tot_acc += accuracy
            # tot_loss += loss_outputs

            vs.append(code_set)
            vf.append(code_f)
            tg.append(target)
            if (batch_idx + 1) % n_batch_verif == 0:
                idx += 1
                vs = torch.stack(vs).flatten(start_dim=0, end_dim=1)
                vf = torch.stack(vf).flatten(start_dim=0, end_dim=1)
                tg = torch.stack(tg).flatten(start_dim=0, end_dim=1)
                Ptp01[idx], Ptp05[idx], Ptp1[idx], AUC[idx] = acc_authentication(model, logisticReg, H0_id_v, H0_data_v,
                                                                                 tg, vf.size(0), vs, vf, m_set,
                                                                                 n_batch_verif)
                vs, vf, tg = [], [], []
    avg_loss = tot_loss / n_batches
    avg_acc = tot_acc / n_batches
    print('Evaluation --->avg_loss: %.3f' % avg_loss, 'avg_acc: %.3f' % avg_acc,
          ' ptp01: %.3f' % np.mean(Ptp01), 'ptp05: %.3f' % np.mean(Ptp05)
          , 'ptp1: %.3f' % np.mean(Ptp1), ' auc: %.3f' % np.mean(AUC))
