import cv2
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.functional as F

import os
import logging
import pprint
import json
from config import config

import _init_paths
from .train import get_sampler


from .auto_helper import *
from datasets.base_dataset import BaseDataset
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
                      cuda_visible_devices=(0,1),
                      batch_size_per_gpu = 4,
                      init_lr = 1e-3,
                      random_seed = 304,
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
    os.environ['MASTER_PORT'] = 6666
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices

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
def train(data_root,record_root,cuda_visible_devices=(0,1)):
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

    # 创建logger并生成输出路径
    AutoTrainer.Creat_Logger(cfg=config, phase='train')
    AutoTrainer.logger.info(config)

    # 保存训练参数到json
    train_param_dict = {'crop_size':crop_size,'nclass':num_class,'backbone':backbone,'net':net}

    with open(os.path.join(AutoTrainer.final_output_dir,"param.json"),"w") as f:
        json.dump(train_param_dict,f)
        print("save param file at %s successfully!"%(AutoTrainer.final_output_dir))

    world_size = 2
    mp.spawn(train_main_worker,nprocs=2,args=(world_size,data_root,record_root,
                                              num_class,crop_size,epoch,backbone,net,pretrained_path,
                                              cuda_visible_devices))
    return AutoTrainer.final_output_dir


def test(data_root,output_root,cuda_visible_devices=(0,1)):
    if not os.path.exists(os.path.join(output_root,'param.json')):
        raise FileNotFoundError('can not find param.json')
    with open(os.path.join(output_root,'param.json'),'r') as f:
        param_dict = json.load(f)
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices

    # 更新test config
    config.GPUS = [0,1]
    config.DATASET.DATASET='custom'
    config.DATASET.NUM_CLASSES = param_dict['nclass']
    config.DATASET.ROOT = data_root

    config.MODEL.NAME = param_dict['net']
    config.MODEL.BACKBONE = param_dict['backbone']

    config.TEST.IMAGE_SIZE = param_dict['crop_size']
    config.TEST.SCALE_LIST = [0.5,1.0,1.5]
    config.TEST.MODEL_FILE = os.path.join(AutoTestor.final_output_dir,'final_state.pth')
    if config.MODEL.NAME == 'seg_hrnet':
        config.MODEL.NUM_OUTPUTS = 1
    else:
        config.MODEL.NUM_OUTPUTS = 2


    AutoTestor.Creat_Logger(output_dir=output_root,phase='test')
    AutoTestor.logger.info(pprint.pformat(config))

    # 设置cudnn
    cudnn.benchmark = True
    cudnn.deterministic = False
    cudnn.enabled = True

    # 创建数据集
    AutoTestor.Build_Dataset(cfg=config)

    # 创建模型
    AutoTestor.Build_Model(cfg=config)

    # 开始评估
    AutoTestor.Run_Testor(cfg=config)

    return AutoTestor.metrics_dict




class InferenceJob(BaseDataset):
    '''
    继承BaseDataset 调用其中multi_scale_inference方法
    '''
    def __init__(self,output_root,cuda_visible_devices=(0,1)):
        self.output_root = output_root
        self.model = None
        self.transform = Compose([ToTensor])

        if not os.path.exists(os.path.join(output_root, 'param.json')):
            raise FileNotFoundError('can not find param.json')
        with open(os.path.join(output_root, 'param.json'), 'r') as f:
            self.param_dict = json.load(f)

        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices
        config.GPUS = [0, 1]

        config.MODEL.NAME = self.param_dict['net']
        config.MODEL.BACKBONE = self.param_dict['backbone']
        config.DATASET.NUM_CLASSES = self.param_dict['nclass']

        config.TEST.SCALE_LIST = [0.5, 1.0, 1.5]
        config.TEST.MODEL_FILE = os.path.join(output_root, 'final_state.pth')
        config.TEST.FLIP_TEST = True
        if config.MODEL.NAME == 'seg_hrnet':
            config.MODEL.NUM_OUTPUTS = 1
        else:
            config.MODEL.NUM_OUTPUTS = 2

        self.cfg = config
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.device = torch.device('cuda:0')


    def _load_model(self):
        AutoTestor.Build_Model(cfg=self.cfg)
        self.model = AutoTestor.model

    def _transform(self,raw_image):
        array = np.asarray(bytearray(raw_image), dtype=np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)
        image = image.unsqueeze(0)
        return image

    def _run(self,raw_image):
        image = self._transform(raw_image)
        image.to(self.device)
        size = image.size()
        pred = self.multi_scale_inference(
            self.cfg,
            self.model,
            image,
            scales=self.cfg.TEST.SCALE_LIST,
            flip=self.cfg.TEST.FLIP_TEST,
            stride_rate=self.cfg.TEST.STRIDE_RATE)
        pred = torch.from_numpy(np.expand_dims(pred, 0))
        pred = pred.permute((0, 3, 1, 2))
        if pred.size()[-2] != size[-2] or pred.size()[-1] != size[-1]:
            pred = F.interpolate(
                pred, size[-2:],
                mode='bilinear', align_corners=True
            )
        if pred.shape[1]==1:
            pred = np.asarray((pred[:,0,:,:]>0).cpu(),dtype = np.uint8)
        else:
            pred = np.asarray(np.argmax(pred.cpu(), axis=1), dtype=np.uint8)

        return pred












































