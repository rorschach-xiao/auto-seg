import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp

import os
import logging
import time
from pathlib import Path
import timeit

import _init_paths

import models
import datasets
from core.criterion import CrossEntropy, OhemCrossEntropy,SoftDiceLoss,LovaszHinge,LovaszSoftmax,LovaszSigmoid,StableBCELoss,ComboLoss
from core.function import train, validate,testval
from utils.utils import create_logger, FullModel
from utils.transform import *


from config import config
from tools.train import get_sampler


class AutoTrainer():
    # 类全局变量
    criterion = None
    model = None
    optimizer = None
    logger=None
    final_output_dir=None
    tb_log_dir=None
    train_dataset=None
    trainloader=None
    val_dataset=None
    valloader=None
    crop_size=None



    def __init__(self):
        pass

    @staticmethod
    def Creat_Logger(cfg,phase='train'):
        root_output_dir = Path(cfg.OUTPUT_DIR)
        # set up logger
        if not root_output_dir.exists():
            print('=> creating {}'.format(root_output_dir))
            root_output_dir.mkdir()

        time_str = time.strftime('%Y-%m-%d-%H-%M')
        dataset = 'custom'+'_'+time_str

        final_output_dir = root_output_dir / dataset
        print('=> creating {}'.format(final_output_dir))
        final_output_dir.mkdir(parents=True, exist_ok=True)


        log_file = '{}_{}.log'.format(time_str, phase)
        final_log_file = final_output_dir / log_file
        head = '%(asctime)-15s %(message)s'
        logging.basicConfig(filename=str(final_log_file),
                            format=head)
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        console = logging.StreamHandler()
        logging.getLogger('').addHandler(console)

        tensorboard_log_dir = Path(cfg.LOG_DIR) / dataset

        print('=> creating {}'.format(tensorboard_log_dir))
        tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
        AutoTrainer.logger = logger
        AutoTrainer.final_output_dir=str(final_output_dir)
        AutoTrainer.tb_log_dir=str(tensorboard_log_dir)

    @staticmethod
    def Build_Dataset(cfg,batch_size,**kwargs):
        train_transform_list = get_train_transform(config)
        train_transform = Compose(train_transform_list)

        train_dataset = eval('datasets.custom')(
            root=cfg.DATASET.ROOT,
            list_path='train.txt',
            num_samples=None,
            num_classes=cfg.DATASET.NUM_CLASSES,
            transform=train_transform)

        train_sampler = get_sampler(train_dataset)
        trainloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=cfg.TRAIN.SHUFFLE and train_sampler is None,
            num_workers=cfg.WORKERS,
            pin_memory=True,
            drop_last=True,
            sampler=train_sampler)

        val_transform_list = get_val_transform(config)
        val_transform = Compose(val_transform_list)
        val_dataset = eval('datasets.custom')(
            root=config.DATASET.ROOT,
            list_path='val.txt',
            num_samples=cfg.TEST.NUM_SAMPLES,
            num_classes=cfg.DATASET.NUM_CLASSES,
            transform=val_transform)

        val_sampler = get_sampler(val_dataset)
        valloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=cfg.WORKERS,
            pin_memory=True,
            sampler=val_sampler)

        AutoTrainer.trainloader=trainloader
        AutoTrainer.train_dataset=train_dataset
        AutoTrainer.valloader = valloader
        AutoTrainer.val_dataset=val_dataset

    @staticmethod
    def Build_Loss(cfg,**kwargs):
        if cfg.LOSS.TYPE == "LOVASZ_HINGE" and cfg.DATASET.NUM_CLASSES == 1:
            AutoTrainer.criterion = LovaszHinge(ignore_label=cfg.TRAIN.IGNORE_LABEL,
                                    weight=AutoTrainer.train_dataset.class_weights,
                                    per_image=True)
        elif cfg.LOSS.TYPE == "LOVASZ_SIGMOID" and cfg.DATASET.NUM_CLASSES == 1:
            AutoTrainer.criterion = LovaszSigmoid(ignore_label=cfg.TRAIN.IGNORE_LABEL,
                                      weight=AutoTrainer.train_dataset.class_weights,
                                      per_image=True)
        elif cfg.LOSS.TYPE == "LOVASZ_SOFTMAX" and cfg.DATASET.NUM_CLASSES > 2:
            AutoTrainer.criterion = LovaszSoftmax(ignore_label=cfg.TRAIN.IGNORE_LABEL,
                                      weight=AutoTrainer.train_dataset.class_weights,
                                      per_image=True)
        elif cfg.LOSS.TYPE == "DICE" and cfg.DATASET.NUM_CLASSES <= 2:
            AutoTrainer.criterion = SoftDiceLoss(weight=AutoTrainer.train_dataset.class_weights, per_image=True)

        elif cfg.LOSS.TYPE == "BCE" and cfg.DATASET.NUM_CLASSES == 1:
            AutoTrainer.criterion = StableBCELoss(ignore_label=cfg.TRAIN.IGNORE_LABEL,
                                      weight=AutoTrainer.train_dataset.class_weights)
        elif cfg.LOSS.TYPE == "COMBO" and cfg.DATASET.NUM_CLASSES == 1:
            AutoTrainer.criterion = ComboLoss(ignore_label=cfg.TRAIN.IGNORE_LABEL
                                  , channel_weight=[0.5, 1], data_weight=AutoTrainer.train_dataset.class_weights)
        elif cfg.LOSS.TYPE == "CE":
            if cfg.LOSS.USE_OHEM:
                AutoTrainer.criterion = OhemCrossEntropy(ignore_label=cfg.TRAIN.IGNORE_LABEL,
                                             thres=cfg.LOSS.OHEMTHRES,
                                             min_kept=cfg.LOSS.OHEMKEEP,
                                             weight=AutoTrainer.train_dataset.class_weights)
            else:
                AutoTrainer.criterion = CrossEntropy(ignore_label=cfg.TRAIN.IGNORE_LABEL,
                                         weight=AutoTrainer.train_dataset.class_weights)
        else:
            raise Exception("Build Failed: only support DiceLoss,CrossEntropy,BCELoss,ComboLoss and LovaszLoss now!")

    @staticmethod
    def Build_Optimizer(cfg,**kwargs):
        def _get_parameter(model):
            params_dict = dict(model.named_parameters())
            if cfg.TRAIN.NONBACKBONE_KEYWORDS:
                bb_lr = []
                nbb_lr = []
                nbb_keys = set()
                for k, param in params_dict.items():
                    if any(part in k for part in cfg.TRAIN.NONBACKBONE_KEYWORDS):
                        nbb_lr.append(param)
                        nbb_keys.add(k)
                    else:
                        bb_lr.append(param)
                print(nbb_keys)
                params = [{'params': bb_lr, 'lr': cfg.TRAIN.LR},
                          {'params': nbb_lr, 'lr': cfg.TRAIN.LR * cfg.TRAIN.NONBACKBONE_MULT}]
            else:
                params = [{'params': list(params_dict.values()), 'lr': cfg.TRAIN.LR}]
            return params

        if cfg.TRAIN.OPTIMIZER == 'sgd':
            params = _get_parameter(AutoTrainer.model)
            optimizer = torch.optim.SGD(params,
                                        lr=cfg.TRAIN.LR,
                                        momentum=cfg.TRAIN.MOMENTUM,
                                        weight_decay=cfg.TRAIN.WD,
                                        nesterov=cfg.TRAIN.NESTEROV,
                                        )
        elif cfg.TRAIN.OPTIMIZER == 'adam':
            params = _get_parameter(AutoTrainer.model)
            optimizer = torch.optim.Adam(params,
                                 lr=cfg.TRAIN.LR,
                                 betas=cfg.TRAIN.BETA,
                                 eps=cfg.TRAIN.EPS,
                                 weight_decay=cfg.TRAIN.WD)
        else:
            raise ValueError('Only Support SGD and Adam optimizer')

        AutoTrainer.optimizer=optimizer

    @staticmethod
    def DDP_Init(cfg,local_rank,world_size,model_name):
        device = torch.device('cuda:{}'.format(local_rank))
        torch.cuda.set_device(device)
        dist.init_process_group(
            backend="nccl", init_method="env://", rank=local_rank, world_size=world_size
        )
        model = eval('models.nets.' + model_name +
                     '.get_seg_model')(cfg)
        model = FullModel(model, AutoTrainer.criterion)
        model = model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            find_unused_parameters=True,
            device_ids=[local_rank],
            output_device=local_rank
        )
        AutoTrainer.model=model

    @staticmethod
    def Find_Crop_Size_And_NClass(dataloader,ignore_label=255,**kwargs):
        avg_h = 0.0
        avg_w = 0.0
        nclass = -1
        print("searching hyper parameters...")
        for idx, batch in enumerate(dataloader):
            image, label, _, _ = batch
            _,ori_h,ori_w,_ = image.shape
            label_uni = np.unique(label)
            nclass = max(nclass,label_uni[-1] if label_uni[-1]!=ignore_label else label_uni[-2]) # update num_class
            avg_h = (avg_h*idx+ori_h)/(idx+1) # update avg_h,avg_w
            avg_w = (avg_w*idx+ori_w)/(idx+1)
        if min(avg_h,avg_w)>600 or max(avg_h,avg_w)>1000:
            crop_size = (520,520)
            aug_type = "crop"
        else:
            crop_size = (int(avg_h),int(avg_w))
            aug_type = "resize"
        AutoTrainer.crop_size=crop_size
        print("searching done!")
        if nclass==1:
            return crop_size,nclass,aug_type
        else:
            aug_type = "crop"
            return crop_size,nclass+1,aug_type

    @staticmethod
    def Find_Epoch(num_class,dataset):
        if num_class == 1:
            if len(dataset) < 1000:
                epoch = 20
            elif len(dataset) < 5000:
                epoch = 40
            else:
                epoch = 80
        else:
            if len(dataset) < 2000:
                epoch = 50
            else:
                epoch = 100
        return epoch

    @staticmethod
    def Find_Network(num_class):
        if num_class==1:
            backbone = 'hrnet18'
            net = 'seg_hrnet'
            pretrained_model_path = 'pretrained_models/hrnetv2_w18_imagenet_pretrained.pth'
        else:
            backbone = 'resnest50'
            net = 'seg_asp_ocr'
            pretrained_model_path = 'pretrained_models/resnest50-528c19ca.pth'
        return backbone,net,pretrained_model_path

    @staticmethod
    def Run_Trainer(local_rank,end_epoch,epoch_iters,lr,writer_dict):
        num_iters = end_epoch * epoch_iters
        best_mIoU = 0
        start = timeit.default_timer()
        for epoch in range(0, end_epoch):
            train(config, epoch, end_epoch,
                  epoch_iters, lr, num_iters,
                  AutoTrainer.trainloader, AutoTrainer.optimizer, AutoTrainer.model, writer_dict)
            valid_loss, mean_IoU, IoU_array = validate(config,
                                                       AutoTrainer.valloader, AutoTrainer.model, writer_dict)


            if local_rank <= 0:

                if mean_IoU > best_mIoU:
                    best_mIoU = mean_IoU
                    torch.save(AutoTrainer.model.module.state_dict(),
                               os.path.join(AutoTrainer.final_output_dir, 'best.pth'))
                msg = 'Loss: {:.3f}, MeanIU: {: 4.4f}, Best_mIoU: {: 4.4f}'.format(
                    valid_loss, mean_IoU, best_mIoU)
                logging.info(msg)
                logging.info(IoU_array)
        if local_rank <=0:
            torch.save(AutoTrainer.model.module.state_dict(),
                       os.path.join(AutoTrainer.final_output_dir, 'final_state.pth'))

            writer_dict['writer'].close()
            end = timeit.default_timer()
            AutoTrainer.logger.info('Hours: %d' % np.int((end - start) / 3600))
            AutoTrainer.logger.info('Done')



class AutoTestor():
    logger=None
    final_output_dir=None
    test_dataset=None
    testloader=None
    model=None
    metrics_dict=None

    def __init__(self):
        pass

    @staticmethod
    def Creat_Logger(output_dir, phase='test'):
        output_dir = Path(output_dir)

        time_str = time.strftime('%Y-%m-%d-%H-%M')

        final_output_dir = output_dir
        print('=> creating {}'.format(final_output_dir))
        final_output_dir.mkdir(parents=True, exist_ok=True)

        log_file = '{}_{}.log'.format(time_str, phase)
        final_log_file = final_output_dir / log_file
        head = '%(asctime)-15s %(message)s'
        logging.basicConfig(filename=str(final_log_file),
                            format=head)
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        console = logging.StreamHandler()
        logging.getLogger('').addHandler(console)

        AutoTestor.logger = logger
        AutoTestor.final_output_dir = str(final_output_dir)

    @staticmethod
    def Build_Dataset(cfg,**kwargs):
        test_transform_list = get_test_transform(cfg)
        test_transform = Compose(test_transform_list)
        test_dataset = eval('datasets.' + cfg.DATASET.DATASET)(
            root=cfg.DATASET.ROOT,
            list_path='testval.txt',
            num_samples=None,
            num_classes=cfg.DATASET.NUM_CLASSES,
            transform=test_transform)

        testloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=cfg.WORKERS,
            pin_memory=True)
        AutoTestor.test_dataset=test_dataset
        AutoTestor.testloader=testloader

    @staticmethod
    def Build_Model(cfg,**kwargs):
        if torch.__version__.startswith('1'):
            module = eval('models.nets.' + cfg.MODEL.NAME)
            module.BatchNorm2d_class = module.BatchNorm2d = torch.nn.BatchNorm2d
            module = eval('models.backbones.basenet')
            module.BatchNorm2d_class = module.BatchNorm2d = torch.nn.BatchNorm2d
            module = eval('models.backbones.hrnet')
            module.BatchNorm2d_class = module.BatchNorm2d = torch.nn.BatchNorm2d
        model = eval('models.nets.' + cfg.MODEL.NAME +
                     '.get_seg_model')(cfg)
        gpus = list(cfg.GPUS)
        model = nn.DataParallel(model, device_ids=gpus).cuda()


        model_state_file = cfg.TEST.MODEL_FILE
        AutoTestor.logger.info('=> loading model from {}'.format(model_state_file))

        pretrained_dict = torch.load(model_state_file)
        model_dict = model.state_dict()
        assert set('module.' + k[6:] for k in pretrained_dict if 'criterion' not in k) == set(model_dict)
        pretrained_dict = {'module.' + k[6:]: v for k, v in pretrained_dict.items()
                           if 'module.' + k[6:] in model_dict.keys()}
        for k, _ in pretrained_dict.items():
            AutoTestor.logger.info(
                '=> loading {} from pretrained model'.format(k))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)

        AutoTestor.model=model

    @staticmethod
    def Run_Testor(cfg,**kwargs):
        start = timeit.default_timer()
        mean_IoU, IoU_array, pixel_acc, mean_acc = testval(cfg,
                                                           AutoTestor.test_dataset,
                                                           AutoTestor.testloader,
                                                           AutoTestor.model,
                                                           sv_dir=AutoTestor.final_output_dir)

        msg = 'MeanIU: {: 4.4f}, Pixel_Acc: {: 4.4f}, \
                    Mean_Acc: {: 4.4f}, Class IoU: '.format(mean_IoU,
                                                            pixel_acc, mean_acc)
        metrics_dict = {'mean_IoU':mean_IoU,'IoU_array':IoU_array,'pixel_acc':pixel_acc,'mean_acc':mean_acc}
        AutoTestor.metrics_dict = metrics_dict

        logging.info(msg)
        logging.info(IoU_array)
        end = timeit.default_timer()
        AutoTestor.logger.info('Mins: %d' % np.int((end - start) / 60))
        AutoTestor.logger.info('Done')











