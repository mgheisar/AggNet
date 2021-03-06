import os
import torchvision
import torch
from AggNet import SetNet, get_clusters, LogisticReg, acc_authentication
from utils_data import BalanceBatchSampler, Reporter
import numpy as np
import yaml
import argparse
from checkpoint import CheckPoint
from history import History
import time
import dill
from vgg_face2 import VGG_Faces2
from itertools import chain
import h5py

# --rn="Run00" --loss="loss_auc_max_v1" --start=0 --nw=2 --m=4 --n_b_train=105 --n_b_valid=35 --n_batch_verif=35 --n_epoch=1000 --u_vgg=5760
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
    exp_name = 'lfw'
    num_workers = args_list['num_workers']
    n_save_epoch = args_list['n_save_epoch']
    n_batches_train = args_list['n_batches_train']
    n_batches_valid = args_list['n_batches_valid']
    upper_vgg = args_list['upper_vgg']
    num_clusters = args_list['num_clusters']
    clustering = args_list['clustering']
    vlad_v2 = args_list['vlad_v2']
    lossFun = args_list['loss']
    optimizer = args_list['optimizer']

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
        std_rgb = np.array([1, 1, 1])
        dataset_validation = torchvision.datasets.ImageFolder(root=dataroot,
                                                         transform=torchvision.transforms.Compose([
                                                             torchvision.transforms.Resize(256),
                                                             torchvision.transforms.CenterCrop(224),
                                                             torchvision.transforms.ToTensor(),
                                                             torchvision.transforms.Normalize(mean=mean_rgb, std=std_rgb)
                                                             ]))
    elif exp_name == 'vgg2':
        validation_dataset_root = '/nfs/nas4/marzieh/marzieh/VGG_Face2/test/'
        dataset_validation = VGG_Faces2(validation_dataset_root, split='validation', upper=upper_vgg)
    #  --------------------------------------------------------------------------------------
    # Batch Sampling: n_samples * n_samples
    #  --------------------------------------------------------------------------------------
    batch_size = n_classes * n_samples
    batch_sampler_t = BalanceBatchSampler(dataset=dataset_train, n_classes=n_classes, n_samples=n_samples,
                                          n_batches_epoch=n_batches_train)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_sampler=batch_sampler_t, num_workers=num_workers)

    batch_sampler_v = BalanceBatchSampler(dataset=dataset_validation, n_classes=n_classes, n_samples=n_samples,
                                          n_batches_epoch=n_batches_valid)
    validation_loader = torch.utils.data.DataLoader(dataset_validation, batch_sampler=batch_sampler_v,
                                                    num_workers=num_workers)
    batch_sampler_H0t = BalanceBatchSampler(dataset=dataset_train, n_classes=n_classes * 2, n_samples=1,
                                            n_batches_epoch=n_batch_verif)
    H0_loader_train = torch.utils.data.DataLoader(dataset_train, batch_sampler=batch_sampler_H0t,
                                                  num_workers=num_workers)
    batch_sampler_H0v = BalanceBatchSampler(dataset=dataset_validation, n_classes=n_classes * 2, n_samples=1,
                                            n_batches_epoch=n_batch_verif)
    H0_loader_validation = torch.utils.data.DataLoader(dataset_validation, batch_sampler=batch_sampler_H0v,
                                                       num_workers=num_workers)
    H0_id_t, H0_data_t, H0_id_v, H0_data_v = [], [], [], []
    dataloader_H0_t = iter(H0_loader_train)
    dataloader_H0_v = iter(H0_loader_validation)
    for i in range(n_batch_verif):
        data = next(dataloader_H0_t)
        H0_id_t.append(data[1])
        H0_data_t.append(data[0])

        data = next(dataloader_H0_v)
        H0_id_v.append(data[1])
        H0_data_v.append(data[0])
    #  --------------------------------------------------------------------------------------
    # Model Definitions
    #  --------------------------------------------------------------------------------------
    model = SetNet(base_model_architecture=model_type, num_clusters=num_clusters, vlad_dim=vlad_dim,
                   vlad_v2=vlad_v2)
    logisticReg = LogisticReg()
    # Initialize NetVLAD
    if clustering:
        get_clusters(dataset_train, num_clusters, model_type=model_type, batch_size=64, n_batches=50000)
    if start:
        initcache = os.path.join(ROOT_DIR, 'centroids',
                                 model_type + '_' + '_' + str(num_clusters) + '_desc_cen.hdf5')
        with h5py.File(initcache, mode='r') as h5:
            clsts = h5.get("centroids")[...]
            traindescs = h5.get("descriptors")[...]
            model.net_vlad.init_params(clsts, traindescs)
            del clsts, traindescs

    # for param in model.base_model.parameters():  # freeze base model
    #     param.requires_grad = False
    model.to(device)
    logisticReg.to(device)
    optimizer_model = torch.optim.SGD(chain(model.parameters(), logisticReg.parameters()),
                                      lr=lr, momentum=0.9, weight_decay=0.001)  # weight_decay=0.001
    model.train()
    logisticReg.train()
    #  --------------------------------------------------------------------------------------
    #  Resume training if start is False
    #  --------------------------------------------------------------------------------------
    if not start:
        reporter = Reporter(ckpt_root=os.path.join(ROOT_DIR, 'ckpt'),
                            exp=exp_name, monitor='acc')
        last_model_filename = reporter.select_last(run=run_name).selected_ckpt
        last_epoch = int(reporter.select_last(run=run_name).last_epoch)
        loss0 = reporter.select_last(run=run_name).last_loss
        loss0 = float(loss0[:-4])
        model.load_state_dict(torch.load(last_model_filename)['model_state_dict'])
        reporter = Reporter(ckpt_root=os.path.join(ROOT_DIR, 'ckpt'),
                            exp=exp_name, monitor='acc')
        last_model_filename = reporter.select_last(run=run_name + '_lr').selected_ckpt
        logisticReg.load_state_dict(torch.load(last_model_filename)['model_state_dict'])
    else:
        last_epoch = -1
        loss0 = 0
    # optimizer_model.load_state_dict(...)
    path_ckpt = '{}/ckpt/{}'.format(ROOT_DIR, exp_name)
    # learning checkpointer
    ckpter = CheckPoint(model=model, optimizer=optimizer_model, path=path_ckpt,
                        prefix=run_name, interval=1, save_num=n_save_epoch, loss0=loss0)
    ckpter_lr = CheckPoint(model=logisticReg, optimizer=optimizer_model, path=path_ckpt,
                           prefix=run_name + '_lr', interval=1, save_num=n_save_epoch, loss0=loss0)
    train_hist = History(name='train_hist' + run_name)
    validation_hist = History(name='validation_hist' + run_name)
    if start:
        # ---------  Training logs before start training -----------------
        model.eval()
        logisticReg.eval()
        with torch.no_grad():
            tot_loss, tot_acc = 0, 0
            n_batches = len(train_loader)
            Ptp01, Ptp05 = np.zeros(n_batches // n_batch_verif), np.zeros(n_batches // n_batch_verif)
            vs, vf, tg = [], [], []
            idx = -1
            for batch_idx, (data, target, img_file, class_id) in enumerate(train_loader):
                data_set = data[np.arange(0, batch_size, n_samples)].to(device)
                data_query = data[np.arange(1, batch_size, n_samples)].to(device)
                v_set = model(data_set, m=m_set)  # single vector per set
                v_f = model(data_query, m=1)  # single vector per query
                Sim = torch.mm(v_set, v_f.t())
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
                    Ptp01[idx], Ptp05[idx] = acc_authentication(model, logisticReg, H0_id_t, H0_data_t,
                                                                  tg, vf.size(0), vs, vf, m_set, n_batch_verif)
                    vs, vf, tg = [], [], []

            avg_loss = tot_loss / n_batches
            avg_acc = tot_acc / n_batches
        print('Training log before start training--->avg_loss: %.3f' % avg_loss,
              'avg_acc: %.3f' % avg_acc, ' ptp01: %.3f' % np.mean(Ptp01), 'ptp05: %.3f' % np.mean(Ptp05))
        train_logs = {'loss': avg_loss, 'acc': avg_acc, 'ptp01': np.mean(Ptp01), 'ptp05': np.mean(Ptp05)}
        train_hist.add(logs=train_logs, epoch=0)
        # ---------  Validation logs before start training -----------------
        model.eval()
        logisticReg.eval()
        tot_loss, tot_acc = 0, 0
        n_batches = len(validation_loader)
        Ptp01, Ptp05 = np.zeros(n_batches // n_batch_verif), np.zeros(n_batches // n_batch_verif)
        vs, vf, tg = [], [], []
        idx = -1
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(validation_loader):
                data_set = data[np.arange(0, batch_size, n_samples)].to(device)
                data_query = data[np.arange(1, batch_size, n_samples)].to(device)
                v_set = model(data_set, m=m_set)  # single vector per set
                v_f = model(data_query, m=1)  # single vector per query
                Sim = torch.mm(v_set, v_f.t())
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

                    Ptp01[idx], Ptp05[idx] = acc_authentication(model, logisticReg, H0_id_v, H0_data_v,
                                                                      tg, vf.size(0), vs, vf, m_set, n_batch_verif)
                    vs, vf, tg = [], [], []
        avg_loss = tot_loss / n_batches
        avg_acc = tot_acc / n_batches
        print('Validation log before start training--->avg_loss: %.3f' % avg_loss, 'avg_acc: %.3f' % avg_acc,
              ' ptp01: %.3f' % np.mean(Ptp01), 'ptp05: %.3f' % np.mean(Ptp05))
        validation_logs = {'loss': avg_loss, 'acc': avg_acc, 'ptp01': np.mean(Ptp01), 'ptp05': np.mean(Ptp05)}
        validation_hist.add(logs=validation_logs, epoch=0)
    else:
        train_hist = dill.load(open(ROOT_DIR + "/ckpt/" + exp_name + train_hist.name + ".pickle", "rb"))
        validation_hist = dill.load(open(ROOT_DIR + "/ckpt/" + exp_name + validation_hist.name + ".pickle", "rb"))
    #  --------------------------------------------------------------------------------------
    # Training
    #  --------------------------------------------------------------------------------------
    for epoch in range(last_epoch + 1, n_epoch):
        print('Training epoch', epoch + 1)
        tot_loss, tot_acc = 0, 0
        n_batches = len(train_loader)
        epoch_time_start = time.time()
        model.train()
        logisticReg.train()
        for batch_idx, (data, target, img_file, class_id) in enumerate(train_loader):
            # data: (batch_size,3,224,224)
            data_set = data[np.arange(0, batch_size, n_samples)].to(device)
            data_query = data[np.arange(1, batch_size, n_samples)].to(device)
            v_set = model(data_set, m=m_set)  # single vector per set
            v_f = model(data_query, m=1)  # single vector per query
            Sim = torch.mm(v_set, v_f.t())  # torch.mm(v_set, v_f.T)
            output = logisticReg(Sim.unsqueeze(-1)).squeeze()
            loss_outputs, accuracy = loss_fn(output, len(v_f), m_set)
            tot_acc += accuracy
            tot_loss += loss_outputs
            optimizer_model.zero_grad()
            # print('device:', loss_outputs.device, 'type:', loss_outputs.dtype, 'value:', loss_outputs)
            loss_outputs.backward()
            optimizer_model.step()
        avg_loss_train = tot_loss / n_batches
        avg_acc_train = tot_acc / n_batches
        #  --------------------------------------------------------------------------------------
        # Validation History
        #  --------------------------------------------------------------------------------------
        # ---------  Validation logs -----------------
        print('Computing Validation logs')
        model.eval()
        logisticReg.eval()
        tot_loss, tot_acc = 0, 0
        n_batches = len(validation_loader)
        Ptp01, Ptp05 = np.zeros(n_batches // n_batch_verif), np.zeros(n_batches // n_batch_verif)
        vs, vf, tg = [], [], []
        idx = -1
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(validation_loader):
                # data: (batch_size,3,224,224)
                data_set = data[np.arange(0, batch_size, n_samples)].to(device)
                data_query = data[np.arange(1, batch_size, n_samples)].to(device)
                v_set = model(data_set, m=m_set)  # single vector per set
                v_f = model(data_query, m=1)  # single vector per query
                Sim = torch.mm(v_set, v_f.t())
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
                    Ptp01[idx], Ptp05[idx] = acc_authentication(model, logisticReg, H0_id_v, H0_data_v,
                                                                      tg, vf.size(0), vs, vf, m_set, n_batch_verif)
                    vs, vf, tg = [], [], []
        avg_loss = tot_loss / n_batches
        avg_acc = tot_acc / n_batches
        print('avg_loss: %.3f' % avg_loss, 'avg_acc: %.3f' % avg_acc, ' --->ptp01: %.3f' % np.mean(Ptp01),
              'ptp05: %.3f' % np.mean(Ptp05))
        validation_logs = {'loss': avg_loss, 'acc': avg_acc, 'ptp01': np.mean(Ptp01), 'ptp05': np.mean(Ptp05)}
        validation_hist.add(logs=validation_logs, epoch=epoch + 1)
        if epoch % 10 == 0:
            print('Computing Training logs')
            model.eval()
            logisticReg.eval()
            tot_loss, tot_acc = 0, 0
            n_batches = len(train_loader)
            Ptp01, Ptp05 = np.zeros(n_batches // n_batch_verif), np.zeros(n_batches // n_batch_verif)
            vs, vf, tg = [], [], []
            idx = -1
            with torch.no_grad():
                for batch_idx, (data, target, img_file, class_id) in enumerate(train_loader):
                    data_set = data[np.arange(0, batch_size, n_samples)].to(device)
                    data_query = data[np.arange(1, batch_size, n_samples)].to(device)
                    v_set = model(data_set, m=m_set)  # single vector per set
                    v_f = model(data_query, m=1)  # single vector per query
                    Sim = torch.mm(v_set, v_f.t())
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
                        Ptp01[idx], Ptp05[idx] = acc_authentication(model, logisticReg, H0_id_t, H0_data_t,
                                                                          tg, vf.size(0), vs, vf, m_set, n_batch_verif)
                        vs, vf, tg = [], [], []
            avg_loss = tot_loss / n_batches
            avg_acc = tot_acc / n_batches
            print('avg_loss: %.3f' % avg_loss, 'avg_acc: %.3f' % avg_acc, ' --->ptp01: %.3f' % np.mean(Ptp01),
                  'ptp05: %.3f' % np.mean(Ptp05))
            train_logs = {'loss': avg_loss, 'acc': avg_acc, 'ptp01': np.mean(Ptp01), 'ptp05': np.mean(Ptp05)}
            train_hist.add(logs=train_logs, epoch=epoch + 1)

        epoch_time_end = time.time()
        if epoch > 0:
            ckpter.last_delete_and_save(epoch=epoch, monitor='acc', loss_acc=validation_logs)
            ckpter_lr.last_delete_and_save(epoch=epoch, monitor='acc', loss_acc=validation_logs)

        ckpter.check_on(epoch=epoch, monitor='acc', loss_acc=validation_logs)
        ckpter_lr.check_on(epoch=epoch, monitor='acc', loss_acc=validation_logs)
        print(
            'Epoch {}:\tAverage Loss: {:.3f}\tAverage Accuracy: {:.3f}\tEpoch Time: {:.3f} hours'.format(
                epoch + 1,
                avg_loss_train, avg_acc_train,
                (epoch_time_end - epoch_time_start) / 3600,
            )
        )
        dill.dump(train_hist, file=open(ROOT_DIR + "/ckpt/" + exp_name + train_hist.name + ".pickle", "wb"))
        dill.dump(validation_hist, file=open(ROOT_DIR + "/ckpt/" + exp_name + validation_hist.name + ".pickle", "wb"))
