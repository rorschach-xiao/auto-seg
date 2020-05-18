import os
import torch

import numpy as np
import cv2
from PIL import Image

from config import config
from .base_dataset import BaseDataset

class Cityscapes(BaseDataset):
    def __init__(self,
                 root,
                 list_path,
                 num_samples=None,
                 num_classes=19,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225],
                 transform = None):
        super(Cityscapes,self).__init__(root,list_path,num_samples,num_classes,mean,std,transform)

        ignore_label = config.TRAIN.IGNORE_LABEL
        self.label_mapping = {-1: ignore_label, 0: ignore_label,
                              1: ignore_label, 2: ignore_label,
                              3: ignore_label, 4: ignore_label,
                              5: ignore_label, 6: ignore_label,
                              7: 0, 8: 1, 9: ignore_label,
                              10: ignore_label, 11: 2, 12: 3,
                              13: 4, 14: ignore_label, 15: ignore_label,
                              16: ignore_label, 17: 5, 18: ignore_label,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
                              25: 12, 26: 13, 27: 14, 28: 15,
                              29: ignore_label, 30: ignore_label,
                              31: 16, 32: 17, 33: 18}
        self.class_weights = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345,
                                                1.0166, 0.9969, 0.9754, 1.0489,
                                                0.8786, 1.0023, 0.9539, 0.9843,
                                                1.1116, 0.9037, 1.0865, 1.0955,
                                                1.0865, 1.1529, 1.0507]).cuda()

    def convert_label(self, label, inverse=False):
        temp = label.copy()
        if inverse:
            for v, k in self.label_mapping.items():
                label[temp == k] = v
        else:
            for k, v in self.label_mapping.items():
                label[temp == k] = v
        return label

    def get_palette(self, n):
        palette = [0] * (n * 3)
        for j in range(0, n):
            lab = j
            palette[j * 3 + 0] = 0
            palette[j * 3 + 1] = 0
            palette[j * 3 + 2] = 0
            i = 0
            while lab:
                palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
                palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
                palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
                i += 1
                lab >>= 3
        return palette

    def __getitem__(self,index):
        item = self.files[index]
        name = item["name"]
        image = cv2.imread(os.path.join(self.root, 'cityscapes', item["img"]),
                           cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        size = image.shape

        if 'test' in self.list_path and 'val' not in self.list_path:
            if self.transform != None:
                image = self.transform(image)
            return image, np.array(size), name

        label = cv2.imread(os.path.join(self.root, 'cityscapes', item["label"]),
                           cv2.IMREAD_GRAYSCALE)
        label = self.convert_label(label)

        if self.transform != None:
            image, label = self.transform(image, label)

        return image, label, np.array(size), name

    def save_pred(self, preds, sv_path, name):
        sv_color_path = os.path.join('/'.join(sv_path.split("/")[:-1]), "color_result")
        palette = self.get_palette(256)
        preds = np.asarray(np.argmax(preds.cpu(), axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            pred = preds[i]  # label from 0 to 18
            save_img = Image.fromarray(pred)
            pred_color = self.convert_label(preds[i], inverse=True)
            save_img.save(os.path.join(sv_path, name[i] + '.png'))
            save_color_img = Image.fromarray(pred_color)
            save_color_img.putpalette(palette)
            save_color_img.save(os.path.join(sv_color_path, name[i] + '.png'))



