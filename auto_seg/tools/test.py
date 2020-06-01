import argparse
import os
import pprint
import shutil
import sys

import logging
import time
import timeit

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import _init_paths
import models
import datasets
from config import config
from config import update_config
from core.function import testval, test
from utils.modelsummary import get_model_summary
from utils.utils import create_logger, FullModel
from utils.transform import *


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args


def main():
    args = parse_args()

    logger, final_output_dir, _ = create_logger(
        config, args.cfg, 'test')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # build model
    if torch.__version__.startswith('1'):
        module = eval('models.nets.' + config.MODEL.NAME)
        module.BatchNorm2d_class = module.BatchNorm2d = torch.nn.BatchNorm2d
        module = eval('models.backbones.basenet')
        module.BatchNorm2d_class = module.BatchNorm2d = torch.nn.BatchNorm2d
        module = eval('models.backbones.hrnet')
        module.BatchNorm2d_class = module.BatchNorm2d = torch.nn.BatchNorm2d
    model = eval('models.nets.' + config.MODEL.NAME +
                 '.get_seg_model')(config)
    gpus = list(config.GPUS)
    model = nn.DataParallel(model, device_ids=gpus).cuda()
    dump_input = torch.rand(
        (1, 3, config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    )
    logger.info(get_model_summary(model.cuda(), dump_input.cuda()))

    if config.TEST.MODEL_FILE:
        model_state_file = config.TEST.MODEL_FILE
    else:
        model_state_file = os.path.join(final_output_dir,
                                        'best.pth')
    logger.info('=> loading model from {}'.format(model_state_file))

    pretrained_dict = torch.load(model_state_file)
    model_dict = model.state_dict()
    assert set('module.'+k[6:] for k in pretrained_dict if 'criterion' not in k) == set(model_dict)
    pretrained_dict = {'module.' + k[6:]: v for k, v in pretrained_dict.items()
                       if 'module.' + k[6:] in model_dict.keys()}
    for k, _ in pretrained_dict.items():
        logger.info(
            '=> loading {} from pretrained model'.format(k))
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=False)

    # prepare data

    test_transform_list = get_test_transform(config)
    test_transform = Compose(test_transform_list)
    test_dataset = eval('datasets.' + config.DATASET.DATASET)(
        root=config.DATASET.ROOT,
        list_path=config.DATASET.TEST_SET,
        num_samples=None,
        num_classes=config.DATASET.NUM_CLASSES,
        transform=test_transform)

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True)

    start = timeit.default_timer()
    if 'val' in config.DATASET.TEST_SET:
        mean_IoU, IoU_array, pixel_acc, mean_acc = testval(config,
                                                           test_dataset,
                                                           testloader,
                                                           model,
                                                           sv_dir=final_output_dir,
                                                           per_img=True)

        msg = 'MeanIU: {: 4.4f}, Pixel_Acc: {: 4.4f}, \
            Mean_Acc: {: 4.4f}, Class IoU: '.format(mean_IoU,
                                                    pixel_acc, mean_acc)
        logging.info(msg)
        logging.info(IoU_array)
    elif 'test' in config.DATASET.TEST_SET:
        test(config,
             test_dataset,
             testloader,
             model,
             sv_dir=final_output_dir)

    end = timeit.default_timer()
    logger.info('Mins: %d' % np.int((end - start) / 60))
    logger.info('Done')


if __name__ == '__main__':
    main()
