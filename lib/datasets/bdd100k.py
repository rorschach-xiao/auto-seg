import os

import cv2
import numpy as np

import torch
from torch.nn import functional as F
from PIL import Image
from .base_dataset import BaseDataset

class BDD100k(BaseDataset):
    def __init__(self,
                 root,
                 list_path,
                 num_samples=None,
                 num_classes=19,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225],
                 transform = None):
        super(BDD100k, self).__init__(root,list_path,num_samples,num_classes,mean,std,transform)

        self.class_weights = torch.FloatTensor([0.8333, 0.9326, 0.8519, 1.0062, 0.9658,
                                                0.9713, 1.0629, 1.0254, 0.8521,0.9658,
                                                0.8416, 1.0429, 1.2143, 0.8717, 0.9688,
                                                0.9980, 1.2477, 1.2021, 1.1456]).cuda()

    def __getitem__(self,index):
        item = self.files[index]
        name = item["name"]
        image = cv2.imread(os.path.join(self.root, item["img"]),cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if 'test' in self.list_path and 'val' not in self.list_path:
            size = image.shape
            if self.transform!=None:
                image = self.transform(image)
            return image, np.array(size), name
        else:
            label = cv2.imread(os.path.join(self.root,
                                            item["label"]),cv2.IMREAD_GRAYSCALE)
            size = label.shape
            if self.transform!=None:
                image,label = self.transform(image,label)
            return image,label,np.array(size),name










