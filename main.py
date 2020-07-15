from __future__ import print_function
import os
import time
from time import strftime
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import admm
import numpy as np
from admm_utils import *
from train import val, val_continuous
from tqdm import tqdm
from torch.optim import lr_scheduler
from model import mainnet
from seg_dynamic import seg_dynamic
from seg_static import seg_static
from dataloader import UAVDatasetTuple
from utils import visualize_sum_testing_result, visualize_sum_testing_result_cont
from correlation import Correlation
from auc import auc
import argparse
from options import parser
args = parser.parse_args()
image_saving_dir = '/home/share_uav/zzh/data/uav_regression/'
image_saving_path = image_saving_dir + args.image_save_folder
#  args.ckpt_dir : checkpoint/ucf101/c3d/channel
args.arch = 'seg_static'
args.admm = False
args.masked_retrain = True
ckpt_name = '{}_{}'.format(args.arch, args.sparsity_type)
args.ckpt_dir = os.path.join('/home/share_uav/zzh/data/checkpoint', ckpt_name)
if args.admm and not args.resume and os.path.exists(args.ckpt_dir):
    i = 1
    while os.path.exists(args.ckpt_dir + '_v{}'.format(i)):
        i += 1
    os.rename(args.ckpt_dir, args.ckpt_dir + '_v{}'.format(i))
os.makedirs(args.ckpt_dir, exist_ok=True)

use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.backends.cudnn.benchmark = True  # will result in non-determinism

kwargs = {'num_workers': args.workers, 'worker_init_fn': np.random.seed(args.seed),
          'pin_memory': True} if use_cuda else {}

''' disable all bag of tricks'''
if args.no_tricks:
    # disable all trick even if they are set to some value
    args.alpha = 0.0

''' working directories '''
dir_profile = 'profile'
device = torch.device("cuda")


def load_multi_gpu(model, checkpoint, optimizer, first=False):
    # baseline model for pruning, pruned model for retrain
    try:
        state_dict = checkpoint['state_dict']
        if not first:
            optimizer.load_state_dict(checkpoint['optimizer'])
    except:
        state_dict = checkpoint
    try:
        model.load_state_dict(state_dict)
    except:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for key, value in state_dict.items():
            newkey = 'module.' + key
            new_state_dict[newkey] = value
        model.load_state_dict(new_state_dict)


def main():

    all_dataset = UAVDatasetTuple(task_label_path=args.data_label_path, init_path=args.init_path,
                                  label_path=args.label_path)

    train_size = int(args.split_ratio * len(all_dataset))
    test_size = len(all_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(all_dataset, [train_size, test_size])
    print("Total image tuples for train: ", len(train_dataset))
    print("Total image tuples for test: ", len(test_dataset))

    print("\nLet's use", torch.cuda.device_count(), "GPUs!\n")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=30,
                                               drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=30,
                                              drop_last=True)

    # model_ft = seg_dynamic()
    model = seg_static()
    # model_ft = mainnet()

    model = nn.DataParallel(model)
    if args.admm and args.masked_retrain:
        raise ValueError('cannot do both masked retrain and admm')

    if use_cuda:
        model.cuda()

    criterion = nn.MSELoss(reduction='sum')
    if args.load_from_main_checkpoint:
        chkpt_mainmodel_path = args.load_from_main_checkpoint
        print("Loading ", chkpt_mainmodel_path)
        model.load_state_dict(torch.load(chkpt_mainmodel_path, map_location=device))

    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.1)

    '''====================='''
    ''' multi-rho admm train'''
    '''====================='''
    initial_rho = args.rho
    if args.admm:
        admm_prune(initial_rho, model, train_loader, test_loader, criterion, optimizer_ft, exp_lr_scheduler)

    '''=============='''
    '''masked retrain'''
    '''=============='''
    if args.masked_retrain:
        masked_retrain(initial_rho, model, train_loader, test_loader, criterion, optimizer_ft, exp_lr_scheduler)


def admm_prune(initial_rho, model, train_loader, test_loader, criterion, optimizer, scheduler):
    current_rho = initial_rho
    # initial_rho 默认值为0.0001
    # current_rho 传递给admm中的rho参数
    # 在admm中rho被用来计算每一层的admm_loss
    # 为什么要进行四次rho的更新？

    load_path = '/home/share_uav/zzh/data/uav_regression/check_point/avg_mainFlow_static_2D/avg_mainFlow_static_2D_epoch_1_162.96567006429038'
    print('>_ Loading baseline/progressive model from {}\n'.format(load_path))

    if args.resume:
        load_path = os.path.join(args.ckpt_dir, '{}_{}.pt'.format(ckpt_name, current_rho))

    if os.path.exists(load_path):
        checkpoint = torch.load(load_path)
    else:
        exit('Checkpoint does not exist.')

    load_multi_gpu(model, checkpoint, optimizer, first=(not args.resume))
    model.cuda()

    start_epoch = 1
    best_loss = np.inf
    best_epoch = 0
    if args.resume:
        start_epoch = checkpoint['epoch'] + 1
        try:
            checkpoint = torch.load(load_path.replace('.pt', '_best.pt'), map_location='cpu')
            best_epoch = checkpoint['epoch']
            best_loss = checkpoint['loss']
        except:
            pass

    # 此处加载的是c3d.yaml文件
    prune_ratio_load_path = 'seg_static.yaml'
    ADMM = admm.ADMM(model, file_name=prune_ratio_load_path, rho=current_rho)
    admm.admm_initialization(args, ADMM=ADMM, model=model)  # intialize Z variable
    # initialization的时候就对model的weight进行了pruning
    if i == 0:
        print('Prune config:')
        for k, v in ADMM.prune_cfg.items():
            print('\t{}: {}'.format(k, v))
        print('')

        shutil.copy(os.path.join(dir_profile, args.config_file + '.yaml'), \
                    os.path.join(args.ckpt_dir, args.config_file + '.yaml'))

    # admm train
    save_path = os.path.join(args.ckpt_dir, '{}_{}.pt'.format(ckpt_name, current_rho))

    for epoch in range(start_epoch, args.epochs + 1):
        print('current rho: {}'.format(current_rho))
        train(ADMM, model, train_loader, criterion, optimizer, scheduler, epoch, args)
        loss, prediction_output, label_output, init_output = val(image_saving_path, model, test_loader,
                                                                 device, criterion, epoch, args.batch_size)

        is_best = loss < best_loss
        save_checkpoint(
            {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': loss,
            },
            is_best, save_path)
        if is_best:
            best_loss = loss
            best_epoch = epoch

        print('Best loss@1: {:.3f}  Best epoch: {}'.format(best_loss, best_epoch))
        print('')

        if args.sparsity_type != 'blkcir' and ((epoch - 1) % args.admm_epochs == 0 or epoch == args.epochs):
            print('Weight < 1e-4:')
            prune_cfg = ADMM.prune_cfg
            # for name in model.state_dict():
            #     # if name not in prune_cfg:
            #     #     continue
            #     print(name)
            for layer in prune_cfg.keys():
                weight = model.state_dict()[layer]
                zeros = len((abs(weight) < 1e-4).nonzero())
                weight_size = torch.prod(torch.tensor(weight.shape))
                print('   {}: {}/{} = {:.4f}'.format(layer.split('module.')[-1], zeros, weight_size,
                                                     float(zeros) / float(weight_size)))
            print('')

    os.rename(save_path.replace('.pt', '_best.pt'), \
              save_path.replace('.pt', '_epoch-{}_loss-{:.3f}.pt'.format(best_epoch, best_loss)))


def masked_retrain(initial_rho, model, train_loader, test_loader, criterion, optimizer, scheduler):
    # load admm trained model
    if not args.resume:
        load_path = os.path.join(args.ckpt_dir, '{}_{}.pt'.format(ckpt_name, initial_rho))
        print('>_ Loading model from {}\n'.format(load_path))
    else:
        load_path = os.path.join(args.ckpt_dir, '{}.pt'.format(ckpt_name))

    if os.path.exists(load_path):
        checkpoint = torch.load(load_path)
    else:
        exit('Checkpoint does not exist.')

    load_multi_gpu(model, checkpoint, optimizer, first=True)
    model.cuda()

    start_epoch = 1
    loss_list = [np.inf]
    best_epoch = 0
    if args.resume:
        start_epoch = checkpoint['epoch'] + 1
        try:
            checkpoint = torch.load(load_path.replace('.pt', '_best.pt'), map_location='cpu')
            best_epoch = checkpoint['epoch']
            best_loss = checkpoint['loss']
        except:
            pass

    # restore scheduler
    for epoch in range(1, start_epoch):
        for _ in range(len(train_loader)):
            scheduler.step()

    config_path = 'seg_static.yaml'
    ADMM = admm.ADMM(model, file_name=config_path, rho=initial_rho)
    print('Prune config:')
    for k, v in ADMM.prune_cfg.items():
        print('\t{}: {}'.format(k, v))
    print('')

    admm.hard_prune(args, ADMM, model)
    epoch_loss_dict = {}

    save_path = os.path.join(args.ckpt_dir, '{}.pt'.format(ckpt_name))
    image_saving_path = image_saving_dir + args.image_save_folder

    for epoch in range(start_epoch, args.epochs + 1):
        idx_loss_dict = train(ADMM, model, train_loader, criterion, optimizer, scheduler, epoch, args)
        loss, prediction_output, label_output, init_output = val(image_saving_path, model, test_loader,
                                                                 device, criterion, epoch, args.batch_size)

        best_loss = min(loss_list)
        is_best = loss < best_loss
        save_checkpoint(
            {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': loss,
            },
            is_best, save_path)
        if is_best:
            best_loss = loss
            best_epoch = epoch

        print('Best loss@1: {:.3f}%  Best epoch: {}\n'.format(best_loss, best_epoch))

        epoch_loss_dict[epoch] = idx_loss_dict
        loss_list.append(loss)

    os.rename(save_path.replace('.pt', '_best.pt'), \
              save_path.replace('.pt', '_epoch-{}_loss-{:.3f}.pt'.format(best_epoch, best_loss)))


def train(ADMM, model, train_loader, criterion, optimizer, scheduler, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    idx_loss_dict = {}

    # switch to train mode
    model.train()

    if args.masked_retrain:
        print('full acc re-train masking')
    elif args.combine_progressive:
        print('progressive admm-train/re-train masking')
    if args.masked_retrain or args.combine_progressive:
        masks = {}
        for name, W in model.named_parameters():
            print(name)
            weight = W.detach()
            non_zeros = weight != 0  # 不为0的parameter是True
            zero_mask = non_zeros.type(torch.float32)  # True被转化为1，False被转化为0， 因此记录下所有为0的parameter的位置
            masks[name] = zero_mask
        print('this is the content of dict masks', masks)

    end = time.time()
    epoch_start_time = time.time()
    k = 0
    for batch_idx, data in enumerate(tqdm(train_loader)):
        # measure data loading time
        data_time.update(time.time() - end)
        optimizer.zero_grad()
        task_label = data['task_label'].to(device).float()
        # Normal
        init = data['init'].to(device).float()

        # print("init shape", init.shape)
        label = data['label'].to(device).float()
        # model prediction
        prediction = model(subx=task_label, mainx=init)

        if args.admm:
            admm.admm_adjust_learning_rate(optimizer, epoch, args)
        else:  # only in masked retrain
            scheduler.step()

        # loss
        loss_mse = criterion(prediction, label.data)

        if args.admm:
            admm.z_u_update(args, ADMM, model, train_loader, optimizer, epoch, input, k)  # update Z and U variables
            loss_mse, admm_loss, mixed_loss = admm.append_admm_loss(args, ADMM, model, loss_mse)  # append admm losss

        losses.update(loss_mse.item(), init.size(0))  # 这里需要注意一下

        # compute gradient and do Adam step
        optimizer.zero_grad()

        if args.admm:
            mixed_loss.backward()
        else:
            loss_mse.backward()

        if args.masked_retrain or args.combine_progressive:
            with torch.no_grad():
                for name, W in model.named_parameters():
                    print(name)
                    if name in masks:
                        # print('now, print the value of masks[{}]'.format(name), masks[name])
                        if name == 'module.bn1d_cat_1.weight':
                            print('W in {module.bn1d_cat_1.weight} is', W)
                        W.grad *= masks[name]  # 将值为0的weight的梯度也清零
                        print('now, print the value of masks[{}]'.format(name), masks[name])

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if k % 5 == 0:
            for param_group in optimizer.param_groups:
                current_lr = param_group['lr']
            print('({0}) lr [{1:.6f}]   '
                  'Epoch [{2}][{3:3d}/{4}]   '
                  'Status [admm-{5}][retrain-{6}]   '
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
                  'Loss {loss.val:.4f} ({loss.avg:.4f})   '
                  .format('Adam', current_lr,
                          epoch, k, len(train_loader), args.admm, args.masked_retrain, batch_time=batch_time,
                          loss=losses))
        if k % 100 == 0:
            idx_loss_dict[k] = losses.avg
        print('[Train] Loss {:.4f}   Time {}'.format(
            losses.avg, int(time.time() - epoch_start_time)))
        k += 1
    return idx_loss_dict


def save_checkpoint(state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        torch.save(state, filename.replace('.pt', '_best.pt'))


if __name__ == '__main__':
    main()
