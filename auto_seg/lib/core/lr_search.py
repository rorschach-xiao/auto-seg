import torch
import numpy as np
import torch.nn as nn
import torch.distributed as dist
from torch import optim

import os
import json
import time
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

import logging
import models
import datasets
from config import update_config
from utils.utils import FullModel
from utils.transform import get_train_transform
from utils.transform import Compose
from core.criterion import *
from core.function import reduce_tensor


def get_smooth(l, smooth_radius=40):
    return [sum(l[max(0, i - smooth_radius):min(len(l), i + smooth_radius + 1)]) / (
            min(len(l), i + smooth_radius + 1) - max(0, i - smooth_radius)) for i in range(len(l))]

class InitLRFindJob(object):

    def __init__(self,config,args,final_output_dir):
        self.base_min_lr = config.TRAIN.BASE_MIN_LR
        self.base_max_lr = config.TRAIN.BASE_MAX_LR
        self.search_epoch = config.TRAIN.SEARCH_EPOCH
        self.args = args
        self.final_output_dir = final_output_dir
        self.model_name = config.MODEL.NAME
        self.batch_size = config.TRAIN.BATCH_SIZE_PER_GPU
        self.device = torch.device('cuda:{}'.format(self.args.local_rank))
        self.dataset,self.dataloader = self._load_dataset(config)
        self.criterion = self._load_criterion(config)
        self.model = self._load_model(config)
        self.search_step = int(len(self.dataset)/(self.batch_size*len(list(config.GPUS))))*self.search_epoch


    def _load_model(self,config):
        '''

        :param config:
        :return: a DDP model for lr-searching
        '''
        model = eval('models.nets.' + config.MODEL.NAME +
                     '.get_seg_model')(config)

        model = FullModel(model, self.criterion)
        model = model.to(self.device)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            find_unused_parameters=True,
            device_ids=[self.args.local_rank],
            output_device=self.args.local_rank
        )
        return model

    def _load_dataset(self,config):



        train_transform_list = get_train_transform(config)
        train_transform = Compose(train_transform_list)
        # train data argumentation
        train_dataset = eval('datasets.' + config.DATASET.DATASET)(
            root=config.DATASET.ROOT,
            list_path=config.DATASET.TRAIN_SET,
            num_samples=None,
            num_classes=config.DATASET.NUM_CLASSES,
            transform=train_transform)
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(train_dataset)
        trainloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=config.TRAIN.SHUFFLE and train_sampler is None,
            num_workers=config.WORKERS,
            pin_memory=True,
            drop_last=True,
            sampler=train_sampler)

        return train_dataset,trainloader

    def _load_criterion(self,config):
        if config.LOSS.TYPE == "LOVASZ_HINGE" and config.DATASET.NUM_CLASSES == 1:
            criterion = LovaszHinge(ignore_label=config.TRAIN.IGNORE_LABEL,
                                    weight=self.dataset.class_weights,
                                    per_image=True)
        elif config.LOSS.TYPE == "LOVASZ_SIGMOID" and config.DATASET.NUM_CLASSES == 1:
            criterion = LovaszSigmoid(ignore_label=config.TRAIN.IGNORE_LABEL,
                                      weight=self.dataset.class_weights,
                                      per_image=True)
        elif config.LOSS.TYPE == "LOVASZ_SOFTMAX" and config.DATASET.NUM_CLASSES > 2:
            criterion = LovaszSoftmax(ignore_label=config.TRAIN.IGNORE_LABEL,
                                      weight=self.dataset.class_weights,
                                      per_image=True)
        elif config.LOSS.TYPE == "DICE" and config.DATASET.NUM_CLASSES <= 2:
            criterion = SoftDiceLoss(weight=self.dataset.class_weights, per_image=True)

        elif config.LOSS.TYPE == "BCE" and config.DATASET.NUM_CLASSES == 1:
            criterion = StableBCELoss(ignore_label=config.TRAIN.IGNORE_LABEL,
                                      weight=self.dataset.class_weights)
        elif config.LOSS.TYPE == "COMBO" and config.DATASET.NUM_CLASSES == 1:
            criterion = ComboLoss(ignore_label=config.TRAIN.IGNORE_LABEL
                                  , channel_weight=[0.5, 1], data_weight=self.train_dataset.class_weights)
        elif config.LOSS.TYPE == "RMI":
            criterion = RMILoss()
        elif config.LOSS.TYPE == "CE":
            if config.LOSS.USE_OHEM:
                criterion = OhemCrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                             thres=config.LOSS.OHEMTHRES,
                                             min_kept=config.LOSS.OHEMKEEP,
                                             weight=self.dataset.class_weights)
            else:
                criterion = CrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                         weight=self.dataset.class_weights)
        else:
            raise Exception("error : only support DiceLoss,CrossEntropy,BCELoss,ComboLoss and LovaszLoss now!")
        return criterion

    def _find_opt_lr(self,lr_record,loss_record):

        if any(np.isnan(np.array(loss_record))):
            nan_pos = np.argmax(np.isnan(loss_record))
            loss_record = loss_record[0:nan_pos]
            lr_record = lr_record[0:nan_pos]

        smooth_loss = get_smooth(loss_record, 40)
        # smooth_loss = []
        # h = 0.0
        # for loss in loss_record:
        #     h = loss * (1 - gamma) + h * gamma
        #     smooth_loss.append(h)

        lr_record = np.array(lr_record)
        loss_record = np.array(smooth_loss)
        model = linear_model.LinearRegression(fit_intercept=True, normalize=True)
        #     model = linear_model.Ridge(alpha=0.001,fit_intercept=True, normalize=True)
        clf = Pipeline([('poly', PolynomialFeatures(degree=6)), ('linear', model)])
        clf.fit(lr_record[:, np.newaxis], loss_record)

        predict_y = clf.predict(lr_record[:, np.newaxis])
        min_pos = 0
        for i in range(1, len(predict_y) - 1):
            if predict_y[i] < predict_y[i - 1] and predict_y[i] < predict_y[i + 1]:
                min_pos = i
                break
        if min_pos == 0 or min_pos == len(predict_y) - 1:
            min_pos = len(predict_y) // 2

        max_lr = lr_record[min_pos] / 4

        return max_lr,list(predict_y),list(smooth_loss)

    def _run_search(self,config):
        start = time.time()
        self.model.train()

        def _get_parameter(model):
            params_dict = dict(model.named_parameters())
            if config.TRAIN.NONBACKBONE_KEYWORDS:
                bb_lr = []
                nbb_lr = []
                nbb_keys = set()
                for k, param in params_dict.items():
                    if any(part in k for part in config.TRAIN.NONBACKBONE_KEYWORDS):
                        nbb_lr.append(param)
                        nbb_keys.add(k)
                    else:
                        bb_lr.append(param)
                params = [{'params': bb_lr, 'lr': self.base_min_lr},
                          {'params': nbb_lr, 'lr': self.base_min_lr * config.TRAIN.NONBACKBONE_MULT}]
            else:
                params = [{'params': list(params_dict.values()), 'lr': self.base_min_lr}]
            return params

        params = _get_parameter(self.model)
        optimizer = optim.SGD(params, lr=self.base_min_lr,
                              momentum=0.9, weight_decay=1e-5)
        loss_record = []
        lr_record = []
        lr_step_factor = pow(self.base_max_lr/self.base_min_lr,1/self.search_step)
        print('start searching learning rate...')
        done = False
        while(not done):
            for i,batch in enumerate(self.dataloader):
                if len(lr_record) >= self.search_step:
                    done=True
                    break
                if len(lr_record) > 200:
                    tmp_loss_record = get_smooth(loss_record)

                if len(lr_record) > 200 and \
                        tmp_loss_record[-1] - tmp_loss_record[0] > 2 and \
                        (tmp_loss_record[-1] - min(tmp_loss_record)) / (tmp_loss_record[0] - min(tmp_loss_record)) > 2:
                    done=True
                    break

                images,labels,_,_ = batch
                images = images.cuda()
                labels = labels.cuda()
                losses,_ = self.model(images,labels)
                loss = losses.mean()
                reduced_loss = reduce_tensor(loss) # 计算不同进程上loss的平均，reduce到cuda:0
                dist.broadcast(reduced_loss,0) # 将平均loss再分发到其他节点

                lr_record.append(optimizer.param_groups[0]['lr'])
                loss_record.append(reduced_loss.item())
                if self.args.local_rank==0:
                    msg = 'Iter: [{}/{}] Lr: {} Loss: {}'.format(len(loss_record),self.search_step,
                                                                 lr_record[-1],reduced_loss)
                    logging.info(msg)

                self.model.zero_grad()
                loss.backward()
                optimizer.step()
                optimizer.param_groups[0]['lr']*=lr_step_factor
                if len(optimizer.param_groups) == 2:
                    optimizer.param_groups[1]['lr'] = optimizer.param_groups[0]['lr']*config.TRAIN.NONBACKBONE_MULT

        del self.model
        torch.cuda.empty_cache()
        opt_lr, regression_loss, smooth_loss = self._find_opt_lr(lr_record, loss_record)
        if self.args.local_rank==0 and len(loss_record)!=0:
            lr_search_param = {'lr_record': lr_record, 'loss_record': loss_record,
                               'regression_loss': regression_loss,'smooth_loss':smooth_loss,'opt_lr': opt_lr}

            with open(os.path.join(self.final_output_dir, 'lr_search_{}_step_{}_{}_param.json'.
                    format(self.search_step,self.base_min_lr,self.base_max_lr)),'w') as f:
                json.dump(lr_search_param,f)
                print('===========> saving lr-searching param successfully!')
        print('rank:{} opt_lr:{}'.format(self.args.local_rank,opt_lr))
        searching_time = time.time()-start
        logging.info('Find optimal lr for %s: %f'%(self.model_name,opt_lr))
        logging.info('searching time:%f'%searching_time)

        return opt_lr
























