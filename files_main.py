from __future__ import print_function
import os
import time
import yaml
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

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


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


def config_read(config):
    if not isinstance(config, str):
        raise Exception('filename must be a str')
    with open(config, 'r') as stream:
        try:
            raw_dict = yaml.full_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return raw_dict['prune_ratios']


def text_save(filename, admm_loss, masked_loss, config_data, admm_layer_states):
    # filename为写入CSV文件的路径，data为要写入数据列表.
    file = open(filename, 'a')
    file.write('configuration: \n')
    for layer in config_data:
        file.write(layer + ' : ' + str(config_data[layer]) + '\n')
    file.write('\n##############################################\n')
    for i in admm_layer_states:
        file.write(i)
    file.write('\n##############################################\n')
    file.write('admm loss list: \n')
    for i in range(len(admm_loss)):
        s = str(admm_loss[i]).replace('[', '').replace(']', '')  # 去除[],这两行按数据不同，可以选择
        s = s.replace("'", '').replace(',', '') + '\n'    # 去除单引号，逗号，每行末尾追加换行符
        if i == 0:
            s = 'best loss: {}'.format(s)
        else:
            s = 'epoch{} loss: {}'.format(i-1, s)
        file.write(s)
    length = len(masked_loss)
    file.write('\n##############################################\n')
    file.write('masked loss list: \n')
    for j in range(length):
        m = str(masked_loss[j]).replace('[', '').replace(']', '')
        m = m.replace("'", '').replace(',', '') + '\n'
        if j == 0:
            m = 'best loss: {}'.format(m)
        elif j == 1:
            m = 'worst loss: {}'.format(m)
        else:
            m = 'epoch{} loss: {}'.format(j - 1, m)
        file.write(m)
    file.close()
    print("Successfully save losses record to " + filename)


args = parser.parse_args()
image_saving_dir = '/home/share_uav/zzh/data/uav_regression/'
image_saving_path = image_saving_dir + args.image_save_folder
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.backends.cudnn.benchmark = True  # will result in non-determinism
kwargs = {'num_workers': args.workers, 'worker_init_fn': np.random.seed(args.seed),
          'pin_memory': True} if use_cuda else {}
device = torch.device("cuda")
args.arch = 'seg_static'
ckpt_name = '{}_{}'.format(args.arch, args.sparsity_type)  # ckpt_name = 'seg_static_filter'


def main():
    for i in range(5):
        config_path = 'seg_static_configs/seg_static{}.yaml'.format(i)
        config_data = config_read(config_path)
        args.admm = True
        args.masked_retrain = False
        args.epochs = 100
        args.ckpt_dir = os.path.join('/home/share_uav/zzh/data/new_checkpoint', ckpt_name)
        if args.admm and not args.resume and os.path.exists(args.ckpt_dir):
            j = 1
            while os.path.exists(args.ckpt_dir + '_v{}'.format(j)):
                j += 1
            # 上次实验存储checkpoint的seg_static_filter文件夹被rename为seg_static_filter_vi
            # 所以每次会新生成一个seg_static_filter文件夹来存储数据
            os.rename(args.ckpt_dir, args.ckpt_dir + '_v{}'.format(j))
        os.makedirs(args.ckpt_dir, exist_ok=True)
        admm_loss_list, admm_layer_states = run(config_path=config_path)
        args.admm = False
        args.masked_retrain = True
        mask_epochs = 80
        masked_loss_list, _ = run(new_epochs=mask_epochs, config_path=config_path)
        txt_directory = os.path.join('/home/share_uav/zzh/data/admm_loss_record/files', ckpt_name)
        os.makedirs(txt_directory, exist_ok=True)
        version = 0
        txt_name = '{}admm_{}mask_losses_record.txt'.format(args.epochs, mask_epochs)
        filename = os.path.join(txt_directory, txt_name)
        while os.path.exists(filename):
            version += 1
            txt_name = '{}admm_{}mask_losses_record_v{}.txt'.format(args.epochs, mask_epochs, version)
            filename = os.path.join(txt_directory, txt_name)
        text_save(filename, admm_loss_list, masked_loss_list, config_data, admm_layer_states)


def run(new_epochs=20, config_path='seg_static.yaml'):
    all_dataset = UAVDatasetTuple(task_label_path=args.data_label_path, init_path=args.init_path,
                                  label_path=args.label_path)

    train_size = int(args.split_ratio * len(all_dataset))
    test_size = len(all_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(all_dataset, [train_size, test_size])

    print('Current pruning type is admm' if args.admm else 'Current pruning type is mask')
    print("Current sparsity type is " + args.sparsity_type)
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
        loss_list, admm_layer_states = admm_prune(initial_rho, model, train_loader, test_loader, criterion, optimizer_ft, exp_lr_scheduler,
                               config_path)
        return loss_list, admm_layer_states

    '''=============='''
    '''masked retrain'''
    '''=============='''
    if args.masked_retrain:
        loss_list, mask_layer_states = masked_retrain(initial_rho, model, train_loader, test_loader, criterion, optimizer_ft,
                                   exp_lr_scheduler, new_epochs, config_path)
        return loss_list, mask_layer_states


def admm_prune(initial_rho, model, train_loader, test_loader, criterion, optimizer, scheduler, config_path):
    current_rho = initial_rho
    layer_states = []
    # initial_rho 默认值为0.0001
    # current_rho 传递给admm中的rho参数
    # 在admm中rho被用来计算每一层的admm_loss

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
    admm_loss_list = [1000]
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

    ADMM = admm.ADMM(model, file_name=config_path, rho=current_rho)
    admm.admm_initialization(args, ADMM=ADMM, model=model)  # intialize Z variable
    # initialization的时候就对model的weight进行了pruning
    print('Prune config:')
    for k, v in ADMM.prune_cfg.items():
        print('\t{}: {}'.format(k, v))
    print('')

    # admm train
    save_path = os.path.join(args.ckpt_dir, '{}_{}.pt'.format(ckpt_name, current_rho))

    for epoch in range(start_epoch, args.epochs + 1):
        print('current rho: {}'.format(current_rho))
        train(ADMM, model, train_loader, criterion, optimizer, scheduler, epoch, args)
        loss, prediction_output, label_output, init_output = val(image_saving_path, model, test_loader,
                                                                 device, criterion, epoch, args.batch_size)

        if loss < admm_loss_list[0]:
            admm_loss_list[0] = loss
        admm_loss_list.append(loss)
        is_best = loss < best_loss

        if is_best:
            best_loss = loss
            best_epoch = epoch
            save_checkpoint(
                {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss': loss,
                },
                is_best, save_path)

        print('Best loss@1: {:.3f}  Best epoch: {}'.format(best_loss, best_epoch))
        print('')

        if args.sparsity_type != 'blkcir' and epoch == args.epochs:
            layer_states.append('Weight < 1e-4:')
            for layer in ADMM.prune_cfg.keys():
                weight = model.state_dict()[layer]
                zeros = len((abs(weight) < 1e-4).nonzero())
                weight_size = torch.prod(torch.tensor(weight.shape))
                # print('   {}: {}/{} = {:.4f}'.format(layer.split('module.')[-1], zeros, weight_size,
                #                                      float(zeros) / float(weight_size)))
                layer_states.append('{}: {}/{} = {:.4f}'.format(layer.split('module.')[-1], zeros, weight_size,
                                                                float(zeros) / float(weight_size)))

    os.rename(save_path.replace('.pt', '_best.pt'), \
              save_path.replace('.pt', '_epoch-{}_loss-{:.3f}.pt'.format(best_epoch, best_loss)))
    return admm_loss_list, layer_states


def masked_retrain(initial_rho, model, train_loader, test_loader, criterion, optimizer, scheduler, epochs, config_path):
    # load admm trained model
    layer_states = []
    if not args.resume:
        load_path = os.path.join(args.ckpt_dir, '{}_{}.pt'.format(ckpt_name, initial_rho))
        print('>_ Loading model from {}\n'.format(load_path))
    else:
        load_path = os.path.join(args.ckpt_dir, '{}.pt'.format(ckpt_name))

    if os.path.exists(load_path):
        checkpoint = torch.load(load_path, map_location=device)
    else:
        exit('Checkpoint does not exist.')

    load_multi_gpu(model, checkpoint, optimizer, first=True)
    model.cuda()
    start_epoch = 1
    mask_loss_list = [1000, 0]
    best_epoch = 0
    if args.resume:
        start_epoch = checkpoint['epoch'] + 1
        try:
            # checkpoint = torch.load(load_path.replace('.pt', '_best.pt'), map_location='cpu')
            checkpoint = torch.load(load_path.replace('.pt', '_best.pt'),
                                    map_location=device)
            best_epoch = checkpoint['epoch']
            best_loss = checkpoint['loss']
        except:
            pass

    # restore scheduler
    for epoch in range(1, start_epoch):
        for _ in range(len(train_loader)):
            scheduler.step()

    ADMM = admm.ADMM(model, file_name=config_path, rho=initial_rho)
    print('Prune config:')
    for k, v in ADMM.prune_cfg.items():
        print('\t{}: {}'.format(k, v))
    print('')

    admm.hard_prune(args, ADMM, model)
    # epoch_loss_dict = {}

    save_path = os.path.join(args.ckpt_dir, '{}.pt'.format(ckpt_name))
    for epoch in range(start_epoch, epochs + 1):
        train(ADMM, model, train_loader, criterion, optimizer, scheduler, epoch, args)
        loss, prediction_output, label_output, init_output = val(image_saving_path, model, test_loader,
                                                                 device, criterion, epoch, args.batch_size)

        if loss < mask_loss_list[0]:
            mask_loss_list[0] = loss
        if loss > mask_loss_list[1]:
            mask_loss_list[1] = loss
        mask_loss_list.append(loss)
        best_loss = np.inf
        is_best = loss < best_loss

        if is_best:
            best_loss = loss
            best_epoch = epoch
            save_checkpoint(
                {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss': loss,
                },
                is_best, save_path)

        print('Best loss@1: {:.3f}  Best epoch: {}\n'.format(best_loss, best_epoch))

        if args.sparsity_type != 'blkcir' and epoch == args.epochs:
            layer_states.append('the rates of Weight equal to 0 in each layer:')
            for layer, W in model.named_parameters():
                if layer in ADMM.prune_cfg.keys():
                    weight = model.state_dict()[layer]
                    zeros = len((abs(weight) == 0).nonzero())
                    weight_size = torch.prod(torch.tensor(weight.shape))
                    layer_states.append('{}: {}/{} = {:.4f}'.format(layer.split('module.')[-1], zeros, weight_size,
                                                                    float(zeros) / float(weight_size)))

    os.rename(save_path.replace('.pt', '_best.pt'), \
              save_path.replace('.pt', '_epoch-{}_loss-{:.3f}.pt'.format(best_epoch, best_loss)))
    return mask_loss_list, layer_states


def train(ADMM, model, train_loader, criterion, optimizer, scheduler, epoch, args):
    print('Current pruning type is admm' if args.admm else 'Current pruning type is mask')
    print("Current sparsity type is " + args.sparsity_type)
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    if args.masked_retrain:
        print('full acc re-train masking')
    elif args.combine_progressive:
        print('progressive admm-train/re-train masking')
    if args.masked_retrain or args.combine_progressive:
        masks = {}
        for name, W in model.named_parameters():
            weight = W.detach()
            non_zeros = weight != 0  # 不为0的parameter是True
            zero_mask = non_zeros.type(torch.float32)  # True被转化为1，False被转化为0， 因此记录下所有为0的parameter的位置
            masks[name] = zero_mask

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

        losses.update(loss_mse.item(), init.size(0))

        # compute gradient and do Adam step
        optimizer.zero_grad()

        if args.admm:
            mixed_loss.backward()
        else:
            loss_mse.backward()

        if args.masked_retrain or args.combine_progressive:
            with torch.no_grad():
                for name, W in model.named_parameters():
                    if name in masks:
                        if W.grad is None:
                            # print('the grad of weight in {} layer is None'.format(name))
                            continue
                        W.grad *= masks[name]  # 将值为0的weight的梯度也清零

        optimizer.step()
        print('[Train] Loss {:.4f}   Time {}'.format(
            losses.avg, int(time.time() - epoch_start_time)))


def save_checkpoint(state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        torch.save(state, filename.replace('.pt', '_best.pt'))


if __name__ == '__main__':
    main()
