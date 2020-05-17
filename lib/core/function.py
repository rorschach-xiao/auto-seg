# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import logging
import os
import time

import numpy as np
import numpy.ma as ma
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import functional as F

from lib.utils.utils import AverageMeter
from lib.utils.utils import get_confusion_matrix
from lib.utils.utils import adjust_learning_rate

import lib.utils.distributed as dist
from .criterion import iou_dice_binary

def reduce_tensor(inp):
    """
    Reduce the loss from all processes so that
    process with rank 0 has the averaged results.
    """
    world_size = dist.get_world_size()
    if world_size < 2:
        return inp
    with torch.no_grad():
        reduced_inp = inp
        torch.distributed.reduce(reduced_inp, dst=0)
    return reduced_inp / world_size


def train(config, epoch, num_epoch, epoch_iters, base_lr,
          num_iters, trainloader, optimizer, model, writer_dict):
    # Training
    model.train()

    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    tic = time.time()
    cur_iters = epoch*epoch_iters
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']

    for i_iter, batch in enumerate(trainloader, 0):
        images, labels, _, _ = batch
        images = images.cuda()
        labels = labels.long().cuda()

        losses, _ = model(images, labels)
        loss = losses.mean()

        if dist.is_distributed():
            reduced_loss = reduce_tensor(loss)
        else:
            reduced_loss = loss

        model.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss
        ave_loss.update(reduced_loss.item())

        lr = adjust_learning_rate(optimizer,
                                  base_lr,
                                  num_iters,
                                  i_iter+cur_iters)

        if i_iter % config.PRINT_FREQ == 0 and dist.get_rank() == 0:
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                  'lr: {}, Loss: {:.6f}' .format(
                      epoch, num_epoch, i_iter, epoch_iters,
                      batch_time.average(), [x['lr'] for x in optimizer.param_groups], ave_loss.average())
            logging.info(msg)

    writer.add_scalar('train_loss', ave_loss.average(), global_steps)
    writer_dict['train_global_steps'] = global_steps + 1

def validate(config, testloader, model, writer_dict):
    model.eval()
    ave_loss = AverageMeter()
    nums = config.MODEL.NUM_OUTPUTS
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES, nums))
    with torch.no_grad():
        ious = [[] for i in range(nums)]
        dices = [[] for i in range(nums)]
        for idx, batch in enumerate(testloader):
            image, label, _, _ = batch
            size = label.size()
            image = image.cuda()
            label = label.long().cuda()

            losses, pred = model(image, label)

            if not isinstance(pred, (list, tuple)):
                pred = [pred]
            for i, x in enumerate(pred):
                x = F.interpolate(
                    input=x, size=size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )
                if config.DATASET.NUM_CLASSES>2:
                    confusion_matrix[..., i] += get_confusion_matrix(
                        label,
                        x,
                        size,
                        config.DATASET.NUM_CLASSES,
                        config.TRAIN.IGNORE_LABEL
                    )
                else:
                    assert x.size()[1]==1
                    pred_ = x[:, 0, :, :] > 0
                    iou_batch, dice_batch,_ = iou_dice_binary(pred_, label)
                    ious[i].append(iou_batch)
                    dices[i].append(dice_batch)

            if idx % 10 == 0:
                print(idx)

            loss = losses.mean()
            if dist.is_distributed():
                reduced_loss = reduce_tensor(loss)
            else:
                reduced_loss = loss
            ave_loss.update(reduced_loss.item())

    if dist.is_distributed():
        confusion_matrix = torch.from_numpy(confusion_matrix).cuda()
        reduced_confusion_matrix = reduce_tensor(confusion_matrix)
        confusion_matrix = reduced_confusion_matrix.cpu().numpy()
    mean_IoUs=[]
    IoU_arrays=[]
    for i in range(nums):
        if config.DATASET.NUM_CLASSES<=2:
            IoU_array = [np.mean(ious[i])]
            mean_IoU = IoU_array[0]
            mean_dice = np.mean(dices[i])
            mean_IoUs.append(mean_IoU)
            IoU_arrays.append(IoU_array)
            logging.info('dice coefficient: %.4f'%(mean_dice))
        else:
            pos = confusion_matrix[..., i].sum(1)
            res = confusion_matrix[..., i].sum(0)
            tp = np.diag(confusion_matrix[..., i])
            IoU_array = (tp / np.maximum(1.0, pos + res - tp))
            mean_IoU = IoU_array.mean()
            mean_IoUs.append(mean_IoU)
            IoU_arrays.append(IoU_array)
        if dist.get_rank() <= 0:
            logging.info('{} {} {}'.format(i, IoU_array, mean_IoU))

    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']
    writer.add_scalar('valid_loss', ave_loss.average(), global_steps)
    writer.add_scalar('valid_mIoU', mean_IoUs[config.TEST.OUTPUT_INDEX], global_steps)
    writer_dict['valid_global_steps'] = global_steps + 1
    return ave_loss.average(), mean_IoUs[config.TEST.OUTPUT_INDEX], IoU_array[config.TEST.OUTPUT_INDEX]


def testval(config, test_dataset, testloader, model,
            sv_dir='', sv_pred=True):
    model.eval()
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))
    with torch.no_grad():
        ious = []
        dices = []
        PAs = []
        for index, batch in enumerate(tqdm(testloader)):
            image, label, _, name = batch
            size = label.size()

            pred = test_dataset.multi_scale_inference(
                config,
                model,
                image,
                scales=config.TEST.SCALE_LIST,
                flip=config.TEST.FLIP_TEST,
                stride_rate=config.TEST.STRIDE_RATE)
            pred = torch.from_numpy(np.expand_dims(pred,0))
            pred = pred.permute((0,3,1,2))

            if pred.size()[-2] != size[-2] or pred.size()[-1] != size[-1]:
                pred = F.interpolate(
                    pred, size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )
            if config.DATASET.NUM_CLASSES>2:
                confusion_matrix += get_confusion_matrix(
                    label,
                    pred,
                    size,
                    config.DATASET.NUM_CLASSES,
                    config.TRAIN.IGNORE_LABEL)
            else :
                assert pred.size()[1] == 1
                out = (pred[:, 0, :, :] > 0).cpu()
                iou_batch, dice_batch,pa = iou_dice_binary(out, label)
                ious.append(iou_batch)
                dices.append(dice_batch)
                PAs.append(pa)


            if sv_pred:
                sv_path = os.path.join(sv_dir, config.TEST.TESTVAL_SAVE_DIR)
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(pred, sv_path, name)

            if index % 100 == 0:
                logging.info('processing: %d images' % index)
                if config.DATASET.NUM_CLASSES>2:
                    pos = confusion_matrix.sum(1)
                    res = confusion_matrix.sum(0)
                    tp = np.diag(confusion_matrix)
                    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
                    mean_IoU = IoU_array.mean()
                else:
                    IoU_array = [np.mean(ious)]
                    mean_IoU = IoU_array[0]
                    mean_dice = np.mean(dices)
                    logging.info('dice: %.4f' % (mean_dice))
                logging.info('mIoU: %.4f' % (mean_IoU))
    if config.DATASET.NUM_CLASSES>2:
        pos = confusion_matrix.sum(1)
        res = confusion_matrix.sum(0)
        tp = np.diag(confusion_matrix)
        pixel_acc = tp.sum()/pos.sum()
        mean_acc = (tp/np.maximum(1.0, pos)).mean()
        IoU_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_IoU = IoU_array.mean()
    else:
        IoU_array = [np.mean(ious)]
        mean_IoU = IoU_array[0]
        mean_dice = np.mean(dices)
        pixel_acc = np.mean(PAs)
        mean_acc = pixel_acc
        logging.info('final_dice: %.4f' % (mean_dice))

    return mean_IoU, IoU_array, pixel_acc, mean_acc


def test(config, test_dataset, testloader, model,
         sv_dir='', sv_pred=True):
    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(tqdm(testloader)):
            image, size, name = batch
            size = size[0]
            pred = test_dataset.multi_scale_inference(
                config,
                model,
                image,
                scales=config.TEST.SCALE_LIST,
                flip=config.TEST.FLIP_TEST,
                stride_rate=config.TEST.STRIDE_RATE)
            pred = torch.from_numpy(np.expand_dims(pred, 0))
            pred = pred.permute((0, 3, 1, 2))

            if pred.size()[-2] != size[0] or pred.size()[-1] != size[1]:
                pred = F.interpolate(
                    pred, size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )

            if sv_pred:
                sv_path = os.path.join(sv_dir, config.TEST.TEST_SAVE_DIR)
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(pred, sv_path, name)
