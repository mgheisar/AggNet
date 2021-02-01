import os
import torchvision
import torch
from utils_data import BalanceBatchSampler, Reporter
import numpy as np
import yaml
import argparse
from vgg_face2 import VGG_Faces2

torch.manual_seed(0)
if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force=True)
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    # print(ROOT_DIR)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
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
    lr = args_list['lr']
    vlad_dim = args_list['vlad_dim']
    run_name = args_list['run_name']
    exp_name = args_list['exp_name']
    num_workers = args_list['num_workers']
    n_save_epoch = args_list['n_save_epoch']
    n_batches_train = args_list['n_batches_train']
    n_batches_valid = args_list['n_batches_valid']
    upper_vgg = args_list['upper_vgg']
    num_clusters = args_list['num_clusters']
    clustering = args_list['clustering']
    vlad_v2 = args_list['vlad_v2']
    lossFun = args_list['loss']

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', '--model', type=str, default=model_type,
                        help='model name (default: "resnet50_128")')
    parser.add_argument('--run_name', '--rn', type=str, default=run_name,
                        help='The name for this run (default: "Run01")')
    parser.add_argument('--num_workers', '--nw', type=int, default=num_workers,
                        help='number of workers for Dataloader (num_workers: 8)')
    parser.add_argument('--start', '--start', type=int, default=1,
                        help='Start from scratch (default: 1)')
    parser.add_argument('--vlad_dim', '--dim', type=int, default=vlad_dim,
                        help='the dimension of vlad descriptors (dim: 128)')
    parser.add_argument('--m_set', '--m', type=int, default=m_set,
                        help='the group size')
    parser.add_argument('--n_batches_train', '--n_b_train', type=int, default=n_batches_train,
                        help='Number of batches per epoch for training')
    parser.add_argument('--n_batches_valid', '--n_b_valid', type=int, default=n_batches_valid,
                        help='Number of batches per epoch for validation')
    parser.add_argument('--n_epoch', '--n_epoch', type=int, default=n_epoch,
                        help='Number of epochs')
    parser.add_argument('--upper_vgg', '--u_vgg', type=int, default=upper_vgg,
                        help='Number of images loaded from VGG-Face2')
    parser.add_argument('--num_clusters', '--n_clusters', type=int, default=num_clusters,
                        help='Number of clusters for VLAD')
    parser.add_argument('--vlad_v2', '--vlad_v2', type=int, default=vlad_v2,
                        help='Use VLAD version 2 (default: 0)')
    parser.add_argument('--clustering', '--clustering', type=int, default=clustering,
                        help='Apply Clustering for initializing NetVLAD (default: 0)')
    parser.add_argument('--n_batch_verif', '--n_batch_verif', type=int,
                        help='Number of batches for verification (default: 8)')
    parser.add_argument('--loss', '--loss', type=str, default=lossFun,
                        help='loss function (default: "loss_bc")')
    parser.add_argument('--lr', '--lr', type=float, default=lr,
                        help='learning rate')

    args = parser.parse_args()
    model_type = args.model_type
    run_name = args.run_name
    num_workers = args.num_workers
    start = args.start
    vlad_dim = args.vlad_dim
    m_set = args.m_set
    n_batches_train = args.n_batches_train
    n_batches_valid = args.n_batches_valid
    upper_vgg = args.upper_vgg
    n_epoch = args.n_epoch
    num_clusters = args.num_clusters
    clustering = args.clustering
    vlad_v2 = args.vlad_v2
    n_batch_verif = args.n_batch_verif
    lossFun = args.loss
    lr = args.lr

    if lossFun == 'loss_bc':
        from loss import loss_bc as loss_fn
    elif lossFun == 'loss_bc_fb':
        from loss import loss_bc_fb as loss_fn
    elif lossFun == 'loss_auc_max_v1':
        from loss import loss_auc_max_v1 as loss_fn
    elif lossFun == 'loss_AUCPRHingeLoss':
        from loss import loss_AUCPRHingeLoss as loss_fn

    if n_batches_train == 0:
        n_batches_train = None
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
    if exp_name == 'lfw':
        mean_rgb = np.array([131.0912, 103.8827, 91.4953])
        dataset_validation = torchvision.datasets.ImageFolder(root=dataroot,
                                                         transform=torchvision.transforms.Compose([
                                                             torchvision.transforms.Resize(256),
                                                             torchvision.transforms.CenterCrop(224),
                                                             torchvision.transforms.Normalize(mean=mean_rgb),
                                                             torchvision.transforms.ToTensor()]))
    elif exp_name == 'vgg2':
        validation_dataset_root = '/nfs/nas4/marzieh/marzieh/VGG_Face2/test/'
        dataset_validation = VGG_Faces2(validation_dataset_root, split='validation', upper=upper_vgg)
    #  --------------------------------------------------------------------------------------
    # Batch Sampling: n_samples * n_samples
    #  --------------------------------------------------------------------------------------
    batch_size = n_classes * n_samples
    batch_sampler_t = BalanceBatchSampler(dataset=dataset_train, n_classes=n_classes, n_samples=n_samples,
                                        n_batches_epoch=n_batches_train)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_sampler=batch_sampler_t,
                                               num_workers=num_workers, pin_memory=True)
    import time
    import multiprocessing
    use_cuda = torch.cuda.is_available()
    core_number = multiprocessing.cpu_count()
    batch_size = 64
    best_num_worker = [0, 0]
    best_time = [99999999, 99999999]
    print('cpu_count =', core_number)

    def loading_time(num_workers, pin_memory):
            kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory} if use_cuda else {}
            train_loader = torch.utils.data.DataLoader(dataset_train, batch_sampler=batch_sampler_t,
                                                       **kwargs)
            # start = time.time()
            end = 0
            for epoch in range(4):
                for batch_idx, (data, target, img_file, class_id) in enumerate(train_loader):
                    if batch_idx == 1:
                        start = time.time()
                    if batch_idx == 100:
                        end += time.time() - start
                        break
                    pass

            print("  Used {} second with num_workers = {}".format(end, num_workers))
            return end-start
    for pin_memory in [False, True]:
            print("While pin_memory =", pin_memory)
            for num_workers in range(0, core_number*2+1, 4):
                current_time = loading_time(num_workers, pin_memory)
                if current_time < best_time[pin_memory]:
                    best_time[pin_memory] = current_time
                    best_num_worker[pin_memory] = num_workers
                else: # assuming its a convex function
                    if best_num_worker[pin_memory] == 0:
                        the_range = []
                    else:
                        the_range = list(range(best_num_worker[pin_memory] - 3, best_num_worker[pin_memory]))
                    for num_workers in (the_range + list(range(best_num_worker[pin_memory] + 1,best_num_worker[pin_memory] + 4))):
                        current_time = loading_time(num_workers, pin_memory)
                        if current_time < best_time[pin_memory]:
                            best_time[pin_memory] = current_time
                            best_num_worker[pin_memory] = num_workers
                    break
    if best_time[0] < best_time[1]:
            print("Best num_workers =", best_num_worker[0], "with pin_memory = False")
    else:
            print("Best num_workers =", best_num_worker[1], "with pin_memory = True")

