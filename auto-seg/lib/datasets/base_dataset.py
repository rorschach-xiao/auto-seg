import os
import cv2
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils import data

from config import config

class BaseDataset(data.Dataset):
    def __init__(self,
                 root,
                 list_path,
                 num_samples=None,
                 num_classes=19,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225],
                 transform=None):
        self.root = root
        self.transform = transform
        self.mean = mean
        self.std = std
        self.num_classes = num_classes
        self.list_path = list_path
        self.img_list = [line.strip().split() for line in open(root + list_path)]
        self.class_weights = None
        self.files = self.read_files()
        if num_samples:
            self.files = self.files[:num_samples]

    def __len__(self):
        return len(self.files)

    def read_files(self):
        files = []
        for item in self.img_list:
            if 'train' in self.list_path:
                image_path, label_path = item
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

    def pad_image(self, image, h, w, size, padvalue):
        pad_image = image.copy()
        pad_h = max(size[0] - h, 0)
        pad_w = max(size[1] - w, 0)
        pad_h_half = int(pad_h/2)
        pad_w_half = int(pad_w/2)
        if pad_h > 0 or pad_w > 0:
            pad_image = cv2.copyMakeBorder(image, pad_h_half, pad_h-pad_h_half, pad_w_half,
                                           pad_w-pad_w_half, cv2.BORDER_CONSTANT,
                                           value=padvalue)

        return pad_image
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
            output = (output[0] + output[1].flip(2)) / 2
        else:
            output = output[0]
        output = output.data.cpu().numpy()
        output = output.transpose(1, 2, 0)
        return output

    def multi_scale_inference(self,config, model, image, scales=[1], flip=False,stride_rate = 2/3):
        batch,_,ori_h,ori_w = image.size()
        crop_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
        assert batch ==1
        image = np.squeeze(image.numpy(),axis=0)
        image = np.transpose(image,(1,2,0)) # (H,W,C)
        prediction = np.zeros((ori_h,ori_w,config.DATASET.NUM_CLASSES),dtype=float)
        mean = [item*255 for item in self.mean]
        std = [item*255 for item in self.std]
        for scale in scales:
            long_size = round(scale*config.TEST.BASE_SIZE)
            new_h = new_w = long_size
            if(ori_h>ori_w):
                new_w = round(ori_w/float(ori_h)*long_size)
            else:
                new_h = round(ori_h/float(ori_w)*long_size)
            img_scale = cv2.resize(image,(new_w,new_h),interpolation=cv2.INTER_LINEAR)
            img_pad = self.pad_image(img_scale,new_h,new_w,crop_size,padvalue=mean)
            pad_h_half = int(max(crop_size[0]-new_h,0)/2)
            pad_w_half = int(max(crop_size[1]-new_w,0)/2)
            new_pad_h, new_pad_w, _ = img_pad.shape
            prediction_crop = np.zeros((new_pad_h, new_pad_w, config.DATASET.NUM_CLASSES), dtype=float)
            count_crop = np.zeros((new_pad_h, new_pad_w), dtype=float)
            stride_h = int(np.ceil(crop_size[0] * stride_rate))
            stride_w = int(np.ceil(crop_size[1] * stride_rate))
            grid_h = int(np.ceil(float(new_pad_h - crop_size[0]) / stride_h) + 1)
            grid_w = int(np.ceil(float(new_pad_w - crop_size[1]) / stride_w) + 1)
            for r in range(grid_h):
                for c in range(grid_w):
                    h0=r*stride_h
                    h1=min(h0+crop_size[0],new_pad_h)
                    h0=h1-crop_size[0]
                    w0=c*stride_w
                    w1=min(w0+crop_size[1],new_pad_w)
                    w0=w1-crop_size[1]
                    image_crop = img_pad[h0:h1, w0:w1].copy()
                    count_crop[h0:h1, w0:w1] += 1
                    prediction_crop[h0:h1, w0:w1, :] += self.inference(config,model, image_crop, mean, std,flip)
            prediction_crop /= np.expand_dims(count_crop, 2)
            prediction_crop = prediction_crop[pad_h_half:pad_h_half + new_h,pad_w_half:pad_w_half + new_w]
            prediction_crop = cv2.resize(prediction_crop, (ori_w, ori_h), interpolation=cv2.INTER_LINEAR)
            if (len(prediction_crop.shape)==2):
                prediction_crop = np.expand_dims(prediction_crop,2)
            prediction+=prediction_crop
        prediction/=len(scales)
        return prediction

    def save_pred(self, preds, sv_path, name):
        if preds.shape[1]==1:
            preds = np.asarray((preds[:,0,:,:]>0).cpu(),dtype = np.uint8)
        else:
            preds = np.asarray(np.argmax(preds.cpu(), axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            pred = preds[i]
            save_img = Image.fromarray(pred)
            save_img.save(os.path.join(sv_path, name[i]+'.png'))











