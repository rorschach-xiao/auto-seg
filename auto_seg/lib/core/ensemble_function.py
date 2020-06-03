import logging
import os
import time

import numpy as np
import numpy.ma as ma
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import functional as F

from utils.utils import AverageMeter
from utils.utils import get_confusion_matrix

from .criterion import iou_dice_binary


def testval_ensemble(config,test_dataset,testloader,model_list,
                     sv_dir='',sv_pred=True, per_img=False):

    ious_per_img = []
    dices_per_img = []
    if config.DATASET.NUM_CLASSES == 1:
        confusion_matrix = np.zeros((2, 2))
    else:
        confusion_matrix = np.zeros(
            (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))
    with torch.no_grad():
        for index, batch in enumerate(tqdm(testloader)):
            image, label, _, name = batch
            size = label.size()
            preds = []
            for model in model_list:
                model.eval()
                pred = test_dataset.multi_scale_inference(
                    config,
                    model,
                    image,
                    scales=config.TEST.SCALE_LIST,
                    flip=config.TEST.FLIP_TEST,
                    stride_rate=config.TEST.STRIDE_RATE)
                pred = torch.from_numpy(np.expand_dims(pred, 0))
                pred = pred.permute((0, 3, 1, 2))
                if pred.size()[-2] != size[-2] or pred.size()[-1] != size[-1]:
                    pred = F.interpolate(
                        pred, size[-2:],
                        mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                    )
                preds.append(pred.cpu())
            ensemble_pred = sum(preds)/len(preds)

            confusion_matrix += get_confusion_matrix(
                label,
                ensemble_pred,
                size,
                config.DATASET.NUM_CLASSES,
                config.TRAIN.IGNORE_LABEL)
            if per_img and config.DATASET.NUM_CLASSES==1:
                _pred = ensemble_pred[:,0,:,:]>0
                iou_batch, dice_batch,_ = iou_dice_binary(_pred, label)
                ious_per_img.append(iou_batch)
                dices_per_img.append(dice_batch)

            if sv_pred:
                sv_path = os.path.join(sv_dir, config.TEST.TESTVAL_SAVE_DIR)
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(ensemble_pred, sv_path, name)

            if index % 100 == 0:
                logging.info('processing: %d images' % index)
                pos = confusion_matrix.sum(1)
                res = confusion_matrix.sum(0)
                tp = np.diag(confusion_matrix)
                IoU_array = (tp / np.maximum(1.0, pos + res - tp))
                mean_IoU = IoU_array.mean()
                if config.DATASET.NUM_CLASSES==1:
                    mean_dice = 2*tp[1]/(pos[1]+res[1])
                    logging.info('dice: %.4f' % (mean_dice))
                    if per_img:
                        mean_dice_per_img = np.mean(dices_per_img)
                        mean_iou_per_img = np.mean(ious_per_img)
                        logging.info('dice_per_img:{}, iou_per_img:{}'
                                     .format(mean_dice_per_img,mean_iou_per_img))
                logging.info('mIoU: %.4f' % (mean_IoU))
    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    pixel_acc = tp.sum() / pos.sum()
    mean_acc = (tp / np.maximum(1.0, pos)).mean()
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()
    if config.DATASET.NUM_CLASSES == 1:
        mean_dice = 2 * tp[1] / (pos[1] + res[1])
        logging.info('final_dice: %.4f' % (mean_dice))
        if per_img:
            mean_dice_per_img = np.mean(dices_per_img)
            mean_iou_per_img = np.mean(ious_per_img)
            logging.info('final_dice_per_img:{}, final_iou_per_img:{}'
                         .format(mean_dice_per_img, mean_iou_per_img))

    return mean_IoU, IoU_array, pixel_acc, mean_acc





def test_ensemble(config,test_dataset,testloader,model_list,
                  sv_dir='',sv_pred=True):
    with torch.no_grad():
        for _, batch in enumerate(tqdm(testloader)):
            image, size, name = batch
            size = size[0]
            preds = []
            for model in model_list:
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

                preds.append(pred.cpu())
            ensemble_pred = sum(preds) / len(preds)
            if sv_pred:
                sv_path = os.path.join(sv_dir, config.TEST.TEST_SAVE_DIR)
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(ensemble_pred, sv_path, name)

