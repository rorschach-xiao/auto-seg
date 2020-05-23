import cv2
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.functional as F

import _init_paths
import os
import logging
import pprint
import json
from config import config
import datasets
from datasets.base_dataset import BaseDataset

from .auto_helper import *
from tensorboardX import SummaryWriter


def train_main_worker(local_rank,
                      world_size,
                      queue,
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
                      dilation = True,
                      ignore_label = 255,
                      atrous_rate = [12,24,36],
                      gpus = [0,1],
                      optimizer = 'adam',
                      **kwargs):
    # 更新config
    config.DATASET.NUM_CLASSES = int(num_class)
    config.DATASET.ROOT = data_root
    config.DATASET.DATASET = 'custom'
    config.MODEL.NAME = model_name
    config.MODEL.BACKBONE = backbone
    if config.MODEL.NAME=='seg_hrnet':
        config.MODEL.NUM_OUTPUTS = 1
        config.TRAIN.NONBACKBONE_KEYWORDS=['last_layer']
    else:
        config.MODEL.NUM_OUTPUTS = 2
        config.TRAIN.NONBACKBONE_KEYWORDS = ['asp_ocr','aux_layer']
    config.MODEL.PRETRAINED = pretrained_path
    config.MODEL.DILATION = dilation
    config.MODEL.ATROUS_RATE = atrous_rate

    config.TRAIN.SHUFFLE = True
    config.TRAIN.IMAGE_SIZE = list(crop_size)
    print(crop_size)
    config.TRAIN.BASE_SIZE = max(config.TRAIN.IMAGE_SIZE)

    config.TRAIN.LR = init_lr
    config.TRAIN.IGNORE_LABEL = ignore_label
    config.TRAIN.END_EPOCH = epoch
    config.TRAIN.OPTIMIZER = optimizer
    config.TRAIN.RANDOM_ANGLE_DEGREE= 20
    config.TRAIN.RANDOM_SCALE_MIN= 0.5
    config.TRAIN.RANDOM_SCALE_MAX= 1.5


    config.TRAIN.TRANS_LIST= ['resize',
                              'random_rotate',
                              'random_blur',
                              'random_hflip',
                              'totensor',
                              'normalize']
    config.VAL.TRANS_LIST= ['resize',
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

    config.OUTPUT_DIR = record_root
    AutoTrainer.Creat_Logger(cfg=config, phase='train')
    AutoTrainer.logger.info(config)

    train_param_dict = {'crop_size': list(crop_size), 'nclass': int(num_class), 'backbone': backbone, 'net': model_name}

    with open(os.path.join(AutoTrainer.final_output_dir, "param.json"), "w") as f:
        json.dump(train_param_dict, f)
        print("save param file at %s successfully!" % (AutoTrainer.final_output_dir))

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
    port = random.randint(1000,5000)
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(port)


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
    if local_rank<=0:
        queue.put(AutoTrainer.final_output_dir)
def train(data_root,record_root,cuda_visible_devices='0,1'):
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices

    train_dataset = eval('datasets.custom')(
        root=data_root,
        list_path='train.txt',
        num_samples=None,
        transform=None)

    trainloader = torch.utils.data.DataLoader(train_dataset,batch_size=1)

    # 确定 crop_size 和 类别数量
    crop_size,num_class = AutoTrainer.Find_Crop_Size_And_NClass(trainloader)
    print(crop_size)

    # 确定训练epoch
    epoch = AutoTrainer.Find_Epoch(num_class=num_class,dataset=train_dataset)

    # 确定网络及backbone
    backbone,net,pretrained_path = AutoTrainer.Find_Network(num_class)

    world_size = 2
    ctx = mp.get_context('spawn')
    queue = ctx.Queue()
    mp.spawn(train_main_worker,nprocs=2,args=(world_size,queue,data_root,record_root,
                                              num_class,crop_size,epoch,backbone,net,pretrained_path,))
    return queue.get(block = False)



def test(data_root,output_root,cuda_visible_devices='0,1'):

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

    config.TEST.IMAGE_SIZE = param_dict['crop_size'][::-1] #(w,h)
    config.TEST.BASE_SIZE = max(config.TEST.IMAGE_SIZE)

    # TODO
    config.TEST.SCALE_LIST = [0.5,1.0,1.5]
    config.TEST.MODEL_FILE = os.path.join(output_root,'final_state.pth')
    if config.MODEL.NAME == 'seg_hrnet':
        config.MODEL.NUM_OUTPUTS = 1
    else:
        config.MODEL.NUM_OUTPUTS = 2


    # 设置cudnn
    cudnn.benchmark = True
    cudnn.deterministic = False
    cudnn.enabled = True
    AutoTestor.Creat_Logger(output_dir=output_root, phase='test')
    AutoTestor.logger.info(pprint.pformat(config))

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
    def __init__(self,output_root,cuda_visible_devices='0,1'):
        self.output_root = output_root
        self.model = None
        self.transform = Compose([ToTensor()])

        if not os.path.exists(os.path.join(output_root, 'param.json')):
            raise FileNotFoundError('can not find param.json')
        with open(os.path.join(output_root, 'param.json'), 'r') as f:
            self.param_dict = json.load(f)

        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices
        config.GPUS = [0, 1]

        config.MODEL.NAME = self.param_dict['net']
        config.MODEL.BACKBONE = self.param_dict['backbone']
        config.DATASET.NUM_CLASSES = self.param_dict['nclass']

        # TODO
        config.TEST.IMAGE_SIZE = self.param_dict['crop_size']
        config.TEST.BASE_SIZE = max(config.TEST.IMAGE_SIZE)
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

        self._load_model()


    def _load_model(self):
        AutoTestor.Creat_Logger(self.output_root, 'test')
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
            pred[pred == 1] = 255
        else:
            pred = np.asarray(np.argmax(pred.cpu(), axis=1), dtype=np.uint8)
        pred = pred.squeeze()
        return pred


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AutoCV semantic segmentation Module')
    sub_parser = parser.add_subparsers(help='Command line controlling training process')

    train_parser = sub_parser.add_parser('train', help='training process')
    train_parser.add_argument('--subcommand', default='train')
    train_parser.add_argument('--dataset_path', type=str, default='')
    train_parser.add_argument('--model_path', type=str, default='')
    train_parser.add_argument('--visible_devices_list', type=str, default='0,1')

    test_parser = sub_parser.add_parser('test', help='test process')
    test_parser.add_argument('--subcommand', default='test')
    test_parser.add_argument('--dataset_path', type=str, default='')
    test_parser.add_argument('--model_path', type=str, default='')
    test_parser.add_argument('--visible_devices_list', type=str, default='0')


    args, unknow = parser.parse_known_args()
    rest_args = list(unknow)

    if 'subcommand' in args:
        if args.subcommand == 'train':
            train(args.dataset_path, args.model_path, args.visible_devices_list)
        elif args.subcommand == "test":
            test(args.dataset_path, args.model_path, args.visible_devices_list)
