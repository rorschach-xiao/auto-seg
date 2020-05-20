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



def train_main_worker(data_root,
               recored_path,
               local_rank,
               world_size,
               epoch = 40,
               backbone = "hrnet18",
               model = "seg_hrnet_ocr",
               init_lr = 1e-3,
               random_seed = 304,
               **kwargs
               ):
    # 设置随机种子
    import random
    print('Seeding with', random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)

    # 设置cudnn
    cudnn.benchmark = True
    cudnn.deterministic = False
    cudnn.enabled = True

    # 设置主进程地址和接口
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '6666'

    # 创建DDP process group
    dist.init_process_group(
        backend="nccl",init_method="env://",rank=local_rank,world_size=world_size
    )

    # 创建模型











