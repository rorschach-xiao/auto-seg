import cv2
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp

import os
import logging
import time
from pathlib import Path
import timeit

from config import config
from core.function import train, validate
from utils.utils import create_logger, FullModel

import _init_paths
from .train import get_sampler
from .auto_helper import *
from tensorboardX import SummaryWriter


def train_main_worker(local_rank,
                      world_size,
                      data_root,
                      record_root,
                      num_class,
                      crop_size = (512,512),
                      epoch = 40,
                      backbone = "hrnet18",
                      model_name = "seg_hrnet",
                      pretrained_path = 'pretrained_models/hrnetv2_w18_imagenet_pretrained.pth',
                      batch_size_per_gpu = 4,
                      init_lr = 1e-3,
                      random_seed = 304,
                      multi_grid = False,
                      dilation = True,
                      ignore_label = 255,
                      atrous_rate = [12,24,36],
                      gpus = [0,1],
                      optimizer = 'adam',
                      **kwargs):
    # 更新config
    config.OUTPUT_DIR = record_root
    config.DATASET.NUM_CLASSES = num_class
    config.DATASET.ROOT = data_root
    config.DATASET.DATASET = 'custom'
    config.MODEL.NAME = model_name
    config.MODEL.BACKBONE = backbone
    if config.MODEL.NAME=='seg_hrnet':
        config.MODEL.NUM_OUTPUTS = 1
    else:
        config.MODEL.NUM_OUTPUTS = 2
    config.MODEL.PRETRAINED = pretrained_path
    config.MODEL.DILATION = dilation
    config.MODEL.MULTI_GRID = multi_grid
    config.MODEL.ATROUS_RATE = atrous_rate

    config.TRAIN.SHUFFLE = True
    config.TRAIN.IMAGE_SIZE = crop_size

    config.TRAIN.LR = init_lr
    config.TRAIN.IGNORE_LABEL = ignore_label
    config.TRAIN.END_EPOCH = epoch
    config.TRAIN.OPTIMIZER = optimizer
    config.TRAIN.RANDOM_ANGLE_DEGREE= 20
    config.TRAIN.RANDOM_SCALE_MIN= 0.5
    config.TRAIN.RANDOM_SCALE_MAX= 1.5
    config.TRAIN.TRANS_LIST= ['random_scale',
                              'random_rotate',
                              'random_blur',
                              'random_hflip',
                              'crop',
                              'totensor',
                              'normalize']
    config.VAL.TRANS_LIST= ['crop',
                            'totensor',
                            'normalize']

    if config.DATASET.NUM_CLASSES>2:
        config.LOSS.TYPE = "CE"
        config.LOSS.USE_OHEM = True
    else:
        assert config.DATASET.NUM_CLASSES==1
        config.LOSS.TYPE = "BCE"

    if  config.MODEL.NUM_OUTPUTS == 2:
        config.LOSS.BALANCE_WEIGHTS = [0.4, 1]

    # 创建logger并生成输出路径
    AutoTrainer.Creat_Logger(cfg=config,phase='train')
    AutoTrainer.logger.info(config)


    writer_dict = {
        'writer': SummaryWriter(AutoTrainer.tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # 设置随机种子
    import random
    print('Seeding with', random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)

    # 设置cudnn
    cudnn.benchmark = True
    cudnn.deterministic = False
    cudnn.enabled = True

    # 设置主进程地址和接口 以及使用GPU序号
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '6666'
    os.environ['CUDA_VISIBLE_DEVICES'] = 0, 1

    # 数据准备
    batch_size = batch_size_per_gpu * len(gpus)
    AutoTrainer.Build_Dataset(
        cfg=config, batch_size=batch_size
    )

    # 损失函数
    AutoTrainer.Build_Loss(cfg=config)

    # 初始化DDP 并 创建模型
    AutoTrainer.DDP_Init(cfg=config,local_rank=local_rank,world_size=world_size,
                                 model_name=config.MODEL.NAME)
    AutoTrainer.logger.info(AutoTrainer.model)
    # 优化器定义
    AutoTrainer.Build_Optimizer(cfg=config)

    # 开始训练
    epoch_iters = np.int(AutoTrainer.train_dataset.__len__() / batch_size)
    AutoTrainer.Run_Trainer(local_rank=local_rank,
                            end_epoch=config.TRAIN.END_EPOCH,
                            epoch_iters=epoch_iters,
                            lr=config.TRAIN.LR,
                            writer_dict=writer_dict,
                            )
def trainer(data_root,record_root):
    train_dataset = eval('datasets.custom')(
        root=data_root,
        list_path='train.txt',
        num_samples=None,
        transform=None)
    trainloader = torch.utils.data.DataLoader(train_dataset,batch_size=1)

    # 确定 crop_size 和 类别数量
    crop_size,num_class = AutoTrainer.Find_Crop_Size_And_NClass(trainloader)

    # 确定训练epoch
    epoch = AutoTrainer.Find_Epoch(num_class=num_class,dataset=train_dataset)

    # 确定网络及backbone
    backbone,net,pretrained_path = AutoTrainer.Find_Network(num_class=num_class)

    world_size = 2
    mp.spawn(train_main_worker,nprocs=2,args=(world_size,data_root,record_root,
                                              num_class,crop_size,epoch,backbone,net,pretrained_path))





































