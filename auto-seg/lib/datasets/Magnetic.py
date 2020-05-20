import os

import cv2
import numpy as np

from PIL import Image
from .base_dataset import BaseDataset

class MagneticDataset(BaseDataset):
    def __init__(self,
                 root,
                 list_path,
                 num_samples=None,
                 num_classes=2,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225],
                 transform = None):
        super(MagneticDataset, self).__init__(root, list_path,num_samples,num_classes, mean, std,transform)

    def resize_image(self, image, label, size):
        image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, size, interpolation=cv2.INTER_NEAREST)
        return image, label
    def __getitem__(self,index):
        item = self.files[index]
        name = item["name"]
        image = cv2.imread(os.path.join(self.root,
                                        'Magnetic/', item["img"]), cv2.IMREAD_COLOR)
        size = image.shape
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        label = cv2.imread(os.path.join(self.root,
                                        'Magnetic/', item["label"]), cv2.IMREAD_GRAYSCALE)
        if self.transform!=None:
            image,label =self.transform(image,label)

            return image,label,np.array(size),name









