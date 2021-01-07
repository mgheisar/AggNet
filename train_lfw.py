import os
import torchvision
import torch
from AggNet import SetNet, loss_fn, acc_authentication
from utils_data import BalanceBatchSampler, Reporter
import numpy as np
import yaml
import argparse
from checkpoint import CheckPoint
from history import History
import time
import dill
from vgg_face2 import VGG_Faces2

torch.manual_seed(0)
if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force=True)
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

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
    n_batches_epoch = args_list['n_batches_epoch']

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
                        help='the dimension of vlad descriptors (dim: 256)')
    parser.add_argument('--m_set', '--m', type=int, default=m_set,
                        help='the group size')
    parser.add_argument('--n_batches_epoch', '--n_batches', type=int, default=n_batches_epoch,
                        help='Number of batches per epoch')
    parser.add_argument('--n_epoch', '--n_epoch', type=int, default=n_epoch,
                        help='Number of epochs')

    args = parser.parse_args()
    model_type = args.model_type
    run_name = args.run_name
    num_workers = args.num_workers
    start = args.start
    vlad_dim = args.vlad_dim
    m_set = args.m_set
    n_batches_epoch = args.n_batches_epoch
    if n_batches_epoch == 0:
        n_batches_epoch = None
    n_epoch = args.n_epoch
    #  --------------------------------------------------------------------------------------
    # Load train dataset
    #  --------------------------------------------------------------------------------------
    exp_name = 'lfw'
    if exp_name == 'lfw':
        RGB_MEAN = [0.485, 0.456, 0.406]  # (0.5, 0.5, 0.5)
        RGB_STD = [0.229, 0.224, 0.225]  # (0.5, 0.5, 0.5)
        dataset_train = torchvision.datasets.ImageFolder(root=dataroot,
                                                         transform=torchvision.transforms.Compose([
                                                             torchvision.transforms.Resize(256),
                                                             torchvision.transforms.RandomCrop(224),
                                                             torchvision.transforms.ToTensor(),
                                                             torchvision.transforms.Normalize(mean=RGB_MEAN,
                                                                                              std=RGB_STD),
                                                         ]))
        dataset_validation = dataset_train
    elif exp_name == 'vgg2':
        # VGG Face2
        training_dataset_root = '/nfs/nas4/marzieh/marzieh/VGG_Face2/train/'
        dataset_train = VGG_Faces2(training_dataset_root, split='train')
        validation_dataset_root = '/nfs/nas4/marzieh/marzieh/VGG_Face2/test/'
        dataset_validation = VGG_Faces2(validation_dataset_root, split='validation')
    #  --------------------------------------------------------------------------------------
    # Batch Sampling: n_samples * n_samples
    #  --------------------------------------------------------------------------------------
    batch_size = n_classes * n_samples
    batch_sampler = BalanceBatchSampler(dataset=dataset_train, n_classes=n_classes, n_samples=n_samples,
                                        n_batches_epoch=n_batches_epoch)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_sampler=batch_sampler, num_workers=num_workers)

    batch_sampler = BalanceBatchSampler(dataset=dataset_validation, n_classes=n_classes, n_samples=n_samples,
                                        n_batches_epoch=n_batches_epoch)
    validation_loader = torch.utils.data.DataLoader(dataset_validation, batch_sampler=batch_sampler,
                                                    num_workers=num_workers)
    batch_sampler = BalanceBatchSampler(dataset=dataset_validation, n_classes=100, n_samples=1,
                                        n_batches_epoch=1)
    H0_loader = torch.utils.data.DataLoader(dataset_validation, batch_sampler=batch_sampler,
                                            num_workers=num_workers)
    #  --------------------------------------------------------------------------------------
    # Model Definitions
    #  --------------------------------------------------------------------------------------
    model = SetNet(base_model_architecture=model_type, num_clusters=8, emb_dim=vlad_dim)

    for param in model.base_model.parameters():  # freeze base model
        param.requires_grad = False
    model.to(device)
    optimizer_model = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.001)
    model.train()
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
    else:
        last_epoch = -1
        loss0 = 0
    path_ckpt = '{}/ckpt/{}'.format(ROOT_DIR, exp_name)
    # learning checkpointer
    ckpter = CheckPoint(model=model, optimizer=optimizer_model, path=path_ckpt,
                        prefix=run_name, interval=1, save_num=n_save_epoch, loss0=loss0)
    train_hist = History(name='train_hist' + run_name)
    validation_hist = History(name='validation_hist' + run_name)
    if start:
        # ---------  Training logs before start training -----------------
        model.eval()
        with torch.no_grad():
            tot_loss = 0
            n_batches_epoch = len(train_loader)
            Ptp01, Ptp05 = np.zeros(n_batches_epoch), np.zeros(n_batches_epoch)
            for batch_idx, (data, target) in enumerate(train_loader):
                if batch_idx > 0:
                    break
                data_set = data[np.arange(0, batch_size, n_samples)].to(device)
                data_query = data[np.arange(1, batch_size, n_samples)].to(device)
                v_set = model(data_set, m=m_set)  # single vector per set
                v_f = model(data_query, m=1)  # single vector per query
                loss_outputs = loss_fn(v_set, v_f, m_set)
                tot_loss += loss_outputs

                H0_batch = next(iter(H0_loader))
                Ptp01[batch_idx], Ptp05[batch_idx] = acc_authentication(model, H0_batch, target, n_classes, v_set, v_f, m_set)

            avg_loss = tot_loss / n_batches_epoch
        train_logs = {'loss': avg_loss, 'acc': np.mean(Ptp01), 'acc05': np.mean(Ptp05)}
        train_hist.add(logs=train_logs, epoch=0)
        # ---------  Validation logs before start training -----------------
        model.eval()
        with torch.no_grad():
            tot_loss = 0
            n_batches_epoch = len(validation_loader)
            Ptp01, Ptp05 = np.zeros(n_batches_epoch), np.zeros(n_batches_epoch)
            for batch_idx, (data, target) in enumerate(validation_loader):
                if batch_idx > 0:
                    break
                data_set = data[np.arange(0, batch_size, n_samples)].to(device)
                data_query = data[np.arange(1, batch_size, n_samples)].to(device)
                v_set = model(data_set, m=m_set)  # single vector per set
                v_f = model(data_query, m=1)  # single vector per query
                loss_outputs = loss_fn(v_set, v_f, m_set)
                tot_loss += loss_outputs

                H0_batch = next(iter(H0_loader))
                Ptp01[batch_idx], Ptp05[batch_idx] = acc_authentication(model, H0_batch, target, n_classes, v_set, v_f, m_set)

            avg_loss = tot_loss / n_batches_epoch
        validation_logs = {'loss': avg_loss, 'acc': np.mean(Ptp01), 'acc05': np.mean(Ptp05)}
        validation_hist.add(logs=validation_logs, epoch=0)
    else:
        train_hist = dill.load(open("ckpt/" + exp_name + train_hist.name + ".pickle", "rb"))
        validation_hist = dill.load(open("ckpt/" + exp_name + validation_hist.name + ".pickle", "rb"))
    #  --------------------------------------------------------------------------------------
    # Training
    #  --------------------------------------------------------------------------------------
    for epoch in range(last_epoch + 1, n_epoch):
        print('epoch', epoch)
        # if epoch == 5:
        #     optimizer_model.param_groups[0]['lr'] = lr / 10
        tot_loss = 0
        n_batches_epoch = len(train_loader)
        epoch_time_start = time.time()
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            print('batch', batch_idx)
            # if batch_idx > 20:
            #     break
            # data: (batch_size,3,224,224)
            data_set = data[np.arange(0, batch_size, n_samples)].to(device)
            data_query = data[np.arange(1, batch_size, n_samples)].to(device)
            v_set = model(data_set, m=m_set)  # single vector per set
            v_f = model(data_query, m=1)  # single vector per query
            loss_outputs = loss_fn(v_set, v_f, m_set)
            tot_loss += loss_outputs
            optimizer_model.zero_grad()
            print('device:', loss_outputs.device, 'type:', loss_outputs.dtype, 'value:', loss_outputs)
            loss_outputs.backward()
            optimizer_model.step()
        avg_loss = tot_loss / n_batches_epoch
        #  --------------------------------------------------------------------------------------
        # Validation History
        #  --------------------------------------------------------------------------------------
        # ---------  Validation logs -----------------
        model.eval()
        tot_loss = 0
        n_batches_epoch = len(validation_loader)
        Ptp01, Ptp05 = np.zeros(n_batches_epoch), np.zeros(n_batches_epoch)
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(validation_loader):
                # data: (batch_size,3,224,224)
                data_set = data[np.arange(0, batch_size, n_samples)].to(device)
                data_query = data[np.arange(1, batch_size, n_samples)].to(device)
                v_set = model(data_set, m=m_set)  # single vector per set
                v_f = model(data_query, m=1)  # single vector per query
                loss_outputs = loss_fn(v_set, v_f, m_set)
                tot_loss += loss_outputs

                H0_batch = next(iter(H0_loader))
                Ptp01[batch_idx], Ptp05[batch_idx] = acc_authentication(model, H0_batch, target, n_classes, v_set, v_f, m_set)

        avg_loss = tot_loss / n_batches_epoch
        validation_logs = {'loss': avg_loss, 'acc': np.mean(Ptp01), 'acc05': np.mean(Ptp05)}
        validation_hist.add(logs=validation_logs, epoch=epoch + 1)

        if epoch % 10 == 0:
            model.eval()
            tot_loss = 0
            n_batches_epoch = len(train_loader)
            Ptp01, Ptp05 = np.zeros(n_batches_epoch), np.zeros(n_batches_epoch)
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(train_loader):
                    data_set = data[np.arange(0, batch_size, n_samples)].to(device)
                    data_query = data[np.arange(1, batch_size, n_samples)].to(device)
                    v_set = model(data_set, m=m_set)  # single vector per set
                    v_f = model(data_query, m=1)  # single vector per query
                    loss_outputs = loss_fn(v_set, v_f, m_set)
                    tot_loss += loss_outputs

                    H0_batch = next(iter(H0_loader))
                    Ptp01[batch_idx], Ptp05[batch_idx] = acc_authentication(model, H0_batch, target, n_classes, v_set, v_f, m_set)
            avg_loss = tot_loss / n_batches_epoch
            train_logs = {'loss': avg_loss, 'acc': np.mean(Ptp01), 'acc05': np.mean(Ptp05)}
            train_hist.add(logs=train_logs, epoch=epoch + 1)

        epoch_time_end = time.time()
        if epoch > 0:
            new_path = ckpter.last_delete_and_save(epoch=epoch, monitor='acc', loss_acc=validation_logs)

        ckpter.check_on(epoch=epoch, monitor='acc', loss_acc=validation_logs)
        print(
            'Epoch {}:\tAverage Loss: {:.4f}\tEpoch Time: {:.3f} hours'.format(
                epoch + 1,
                avg_loss,
                (epoch_time_end - epoch_time_start) / 3600,
            )
        )
        dill.dump(train_hist, file=open("ckpt/" + exp_name + train_hist.name + ".pickle", "wb"))
        dill.dump(validation_hist, file=open("ckpt/" + exp_name + validation_hist.name + ".pickle", "wb"))
