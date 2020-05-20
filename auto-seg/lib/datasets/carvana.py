import os

import cv2
import numpy as np

from PIL import Image
from .base_dataset import BaseDataset

class CarvanaDataset(BaseDataset):
    def __init__(self,
                 root,
                 list_path,
                 num_samples=None,
                 num_classes=2,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225],
                 transform = None):
        super(CarvanaDataset, self).__init__(root,list_path,num_samples,num_classes,mean,std,transform)

    def __getitem__(self,index):
        item = self.files[index]
        name = item["name"]
        image = cv2.imread(os.path.join(self.root,
                                        'carvana/', item["img"]), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if 'test' in self.list_path:
            size = image.shape
            if self.transform !=None:
                image = self.transform(image)
            return image, np.array(size), name
        else:
            label = Image.open(os.path.join(self.root,
                                          'carvana/',item["label"]))
            label = np.array(label)
            size = label.shape
            if self.transform:
                image,label = self.transform(image,label)
            return image,label,np.array(size),name









