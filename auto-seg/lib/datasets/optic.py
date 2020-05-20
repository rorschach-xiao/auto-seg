import os

import cv2
import numpy as np

import torch
from torch.nn import functional as F
from PIL import Image
from .base_dataset import BaseDataset

class OPTIC(BaseDataset):
    def __init__(self,
                 root,
                 list_path,
                 num_samples=None,
                 num_classes=2,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225],
                 transform=None):
        super(OPTIC, self).__init__(root, list_path,num_samples,num_classes, mean, std,transform)

        self.class_weights = None

    def __getitem__(self,index):
        item = self.files[index]
        name = item["name"]
        image = cv2.imread(os.path.join(self.root,
                                        'optic_disc_seg/', item["img"]), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if 'test' in self.list_path and 'val' not in self.list_path:
            size = image.shape
            if self.transform !=None:
                image = self.transform(image) # ToTensor
            return image, np.array(size), name
        else:
            label = cv2.imread(os.path.join(self.root,
                                            'optic_disc_seg',item["label"]),cv2.IMREAD_GRAYSCALE)
            label[label==38]=1
            size = label.shape
            if self.transform!=None:
                image,label = self.transform(image,label)
            return image,label,np.array(size),name









