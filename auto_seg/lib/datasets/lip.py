
import os

import cv2
import numpy as np

import torch
from torch.nn import functional as F

from .base_dataset import BaseDataset
from config import config


class LIP(BaseDataset):
    def __init__(self,
                 root,
                 list_path,
                 num_samples=None,
                 num_classes=20,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225],
                 transform = None):

        super(LIP, self).__init__(root, list_path,num_samples,num_classes, mean, std,transform)
        self.flip = True
        self.crop_size = (config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])

    def resize_image(self, image, label, size):
        image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, size, interpolation=cv2.INTER_NEAREST)
        return image, label
    def read_files(self):
        files = []
        for item in self.img_list:
            if 'train' in self.list_path and 'val' not in self.list_path:
                image_path, label_path,_,_ = item
                name = os.path.splitext(os.path.basename(label_path))[0]
                sample = {
                    "img": image_path,
                    "label": label_path,
                    "name": name,
                }
            elif 'val' in self.list_path:
                image_path, label_path = item
                name = os.path.splitext(os.path.basename(label_path))[0]
                sample = {"img": image_path,
                          "label": label_path,
                          "name": name, }
            elif 'test' in self.list_path:
                image_path = item[0]
                name = os.path.splitext(os.path.basename(image_path))[0]
                sample = {"img": image_path,
                          "name": name, }
            else:
                raise NotImplementedError('Unknown subset.')
            files.append(sample)
        return files
    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
        if 'test' in self.list_path and 'val' not in self.list_path:
            image = cv2.imread(os.path.join(self.root, 'LIP/', item["img"]),cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            size = image.shape

            if self.transform is not None:
                image = self.transform(image)
            return image, np.array(size), name

        image = cv2.imread(os.path.join(
            self.root, 'LIP/TrainVal_images/', item["img"]),
            cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = cv2.imread(os.path.join(
            self.root, 'LIP/TrainVal_parsing_annotations/',
            item["label"]),
            cv2.IMREAD_GRAYSCALE)
        size = label.shape
        image, label = self.resize_image(image, label, self.crop_size)


        if self.flip:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, ::flip, :]
            label = label[:, ::flip]

            if flip == -1:
                right_idx = [15, 17, 19]
                left_idx = [14, 16, 18]
                for i in range(0, 3):
                    right_pos = np.where(label == right_idx[i])
                    left_pos = np.where(label == left_idx[i])
                    label[right_pos[0], right_pos[1]] = left_idx[i]
                    label[left_pos[0], left_pos[1]] = right_idx[i]
        if self.transform is not None:
            image,label = self.transform(image.copy(),label.copy())


        return image, label, np.array(size), name



    def inference(self,config,model,image,mean,std,flip=True):
        image = torch.from_numpy(image.transpose((2, 0, 1))).float()
        if std is None:
            for t, m in zip(image, mean):
                t.sub_(m)
        else:
            for t, m, s in zip(image, mean, std):
                t.sub_(m).div_(s)
        image = image.unsqueeze(0).cuda()
        if flip:
            image = torch.cat([image, image.flip(3)], 0)
        with torch.no_grad():
            output = model(image)
        if config.MODEL.NUM_OUTPUTS > 1:
            output = output[config.TEST.OUTPUT_INDEX]
        _, _, h_i, w_i = image.shape
        _, _, h_o, w_o = output.shape
        if (h_o != h_i) or (w_o != w_i):
            output = F.interpolate(output, (h_i, w_i), mode='bilinear', align_corners=True)
        if config.DATASET.NUM_CLASSES>1:  # multi-class
            output = F.softmax(output, dim=1)
        if flip:
            flip_output = output[1]
            flip_output = flip_output.cpu()
            flip_pred = flip_output.cpu().numpy().copy()
            flip_pred[14, :, :] = flip_output[15, :, :]
            flip_pred[15, :, :] = flip_output[14, :, :]
            flip_pred[16, :, :] = flip_output[17, :, :]
            flip_pred[17, :, :] = flip_output[16, :, :]
            flip_pred[18, :, :] = flip_output[19, :, :]
            flip_pred[19, :, :] = flip_output[18, :, :]
            flip_pred = torch.from_numpy(
                flip_pred[:, :, ::-1].copy()).cuda()
            output = (output[0] + flip_pred) / 2
        else:
            output = output[0]
        output = output.data.cpu().numpy()
        output = output.transpose(1, 2, 0)
        return output

