import argparse
import os
import pprint
import shutil
import sys

import logging
import time
import timeit
from pathlib import Path


import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import _init_paths
import models
import datasets
from config import config

from core.ensemble_function import testval_ensemble, test_ensemble
from utils.modelsummary import get_model_summary
from utils.utils import create_logger, FullModel
from utils.transform import *


def update_config(cfg,new_config):
    cfg.merge_from_file(new_config)




class ModelEnsemble():
    logger = None
    final_output_dir = None
    def __init__(self,cfg1,cfg2,data_list,model_path1,model_path2,scale_list=[1.0]):

        cudnn.benchmark = True
        cudnn.deterministic = False
        cudnn.enabled = True
        config.defrost()
        self.data_list = data_list

        update_config(config, cfg1)
        # 创建日志文件夹
        ModelEnsemble._creat_logger(config)
        ModelEnsemble.logger.info(pprint.pformat(config))

        # 加载数据集
        self.test_dataset , self.testloader = self._load_dataset(config)

        # 加载模型
        ModelEnsemble.logger.info('loading No.1 model...')
        config.TEST.MODEL_FILE = model_path1
        self._model_1 = ModelEnsemble._load_model(config,0)

        ModelEnsemble.logger.info('loading No.1 model successfully!')

        update_config(config, cfg2)
        ModelEnsemble.logger.info(pprint.pformat(config))
        config.TEST.MODEL_FILE = model_path2
        ModelEnsemble.logger.info('loading No.2 model...')
        self._model_2 = ModelEnsemble._load_model(config,1)
        ModelEnsemble.logger.info('loading No.2 model successfully!')

        config.TEST.SCALE_LIST = scale_list
        config.TEST.FLIP_TEST = True

        config.freeze()

        self._model_list = [self._model_1,self._model_2]


    @staticmethod
    def _creat_logger(cfg):
        root_output_dir = Path(cfg.OUTPUT_DIR)

        if not root_output_dir.exists():
            print('=> creating {}'.format(root_output_dir))
            root_output_dir.mkdir()

        time_str = time.strftime('%Y-%m-%d-%H-%M')
        dataset = cfg.DATASET.DATASET
        dir_name = 'ensemble'+'_'+time_str

        final_output_dir = root_output_dir / dataset /dir_name
        print('=> creating {}'.format(final_output_dir))
        if not final_output_dir.exists():
            final_output_dir.mkdir(parents=True, exist_ok=True)

        log_file = 'ensemble_{}.log'.format(time_str)
        final_log_file = final_output_dir / log_file
        head = '%(asctime)-15s %(message)s'
        logging.basicConfig(filename=str(final_log_file),
                            format=head)
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        console = logging.StreamHandler()
        logging.getLogger('').addHandler(console)

        ModelEnsemble.logger = logger
        ModelEnsemble.final_output_dir = final_output_dir

    @staticmethod
    def _load_model(cfg,device_id):
        if torch.__version__.startswith('1'):
            module = eval('models.nets.' + cfg.MODEL.NAME)
            module.BatchNorm2d_class = module.BatchNorm2d = torch.nn.BatchNorm2d
            module = eval('models.backbones.basenet')
            module.BatchNorm2d_class = module.BatchNorm2d = torch.nn.BatchNorm2d
            module = eval('models.backbones.hrnet')
            module.BatchNorm2d_class = module.BatchNorm2d = torch.nn.BatchNorm2d
        model = eval('models.nets.' + cfg.MODEL.NAME +
                     '.get_seg_model')(cfg)
        print(model)
        model = model.to('cuda:{}'.format(device_id))
        model_state_file = cfg.TEST.MODEL_FILE
        ModelEnsemble.logger.info('=> loading model from {}'.format(model_state_file))

        pretrained_dict = torch.load(model_state_file)
        model_dict = model.state_dict()
        # pretrained_dict_1 = {'model.pretrained.'+k[6:]: v for k, v in pretrained_dict.items()
        #                   if  'pretrained.'+k[6:] in model_dict.keys() and 'criterion' not in k}
        # pretrained_dict_2 = {'model.ocr.'+k[6:]: v for k, v in pretrained_dict.items()
        #                   if  'ocr' in k or 'cls_head' in k and 'criterion' not in k}
        # pretrained_dict_3 = {'model.'+k[6:]: v for k, v in pretrained_dict.items()
        #                   if  'aux_head' in k and 'criterion' not in k}
        # pretrained_dict = {**pretrained_dict_1,**pretrained_dict_2,**pretrained_dict_3}

        print(set(model_dict)-set(k[6:] for k in pretrained_dict if 'criterion' not in k))
        print(set(k[6:] for k in pretrained_dict if 'criterion' not in k) - set(model_dict))
        assert set(k[6:] for k in pretrained_dict if 'criterion' not in k) == set(model_dict)
        pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                          if  k[6:] in model_dict.keys()}
        # torch.save(pretrained_dict,model_state_file)

        for k, _ in pretrained_dict.items():
                ModelEnsemble.logger.info(
                '=> loading {} from pretrained model'.format(k))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)

        return model


    def _load_dataset(self,cfg):
        test_transform_list = get_test_transform(cfg)
        test_transform = Compose(test_transform_list)
        test_dataset = eval('datasets.' + cfg.DATASET.DATASET)(
            root=cfg.DATASET.ROOT,
            list_path=self.data_list,
            num_samples=None,
            num_classes=cfg.DATASET.NUM_CLASSES,
            transform=test_transform)

        testloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=cfg.WORKERS,
            pin_memory=True)


        return test_dataset,testloader

    def _run_ensemble(self):
        start = timeit.default_timer()
        if 'val' in self.data_list:
            mean_IoU, IoU_array, pixel_acc, mean_acc = testval_ensemble(config,
                                                               self.test_dataset,
                                                               self.testloader,
                                                               self._model_list,
                                                               sv_dir=ModelEnsemble.final_output_dir,
                                                               per_img=True)

            msg = 'MeanIU: {: 4.4f}, Pixel_Acc: {: 4.4f}, \
                Mean_Acc: {: 4.4f}, Class IoU: '.format(mean_IoU,
                                                        pixel_acc, mean_acc)
            logging.info(msg)
            logging.info(IoU_array)
        elif 'test' in self.data_list:
            test_ensemble(config,
                 self.test_dataset,
                 self.testloader,
                 self._model_list,
                 sv_dir=ModelEnsemble.final_output_dir)

        end = timeit.default_timer()
        ModelEnsemble.logger.info('Mins: %d' % np.int((end - start) / 60))
        ModelEnsemble.logger.info('Done')

def parse_args():
    parser = argparse.ArgumentParser(description='Model Ensemble')

    parser.add_argument('--cfg1',
                        help='No.1 model experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('--cfg2',
                        help='No.2 model experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('--data_list',
                        help='testval or test data list',
                        required=True,
                        type=str)
    parser.add_argument('--model_path1',
                        help='No.1 model checkpoint path',
                        required=True,
                        type=str)
    parser.add_argument('--model_path2',
                        help='No.2 model checkpoint path',
                        required=True,
                        type=str)
    parser.add_argument('--scale_list',
                        help='TTA scale list',
                        default='1.0,',
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()


    return args

if __name__ =='__main__':
    args = parse_args()
    cfg1 = args.cfg1
    cfg2 = args.cfg2
    data_list = args.data_list
    model_path1 = args.model_path1
    model_path2 = args.model_path2
    scale_list = [float(scale) for scale in args.scale_list.strip().split(',')]
    model_ensemble = ModelEnsemble(cfg1,cfg2,data_list,model_path1,model_path2,scale_list)
    model_ensemble._run_ensemble()









