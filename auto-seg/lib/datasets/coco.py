import os
import torch

import numpy as np
import cv2
from PIL import Image

from config import config
from .base_dataset import BaseDataset

class CocoDataset(BaseDataset):
    def __init__(self,
                 root,
                 list_path,
                 num_samples=None,
                 num_classes=171,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225],
                 transform=None
                 ):
        super(CocoDataset, self).__init__(root, list_path,num_samples,num_classes, mean, std,transform)
        self._coco2voc_map = {0: 15, 1: 2, 2: 7, 3: 14, 4: 1, 5: 6, 6: 19, 7: 0, 8: 4, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0,
                              14: 0, 15: 3, 16: 8, 17: 12, 18: 13,
                              19: 17, 20: 10, 21: 0, 22: 0, 23: 0, 24: 0, 25: 0, 26: 0, 27: 0, 28: 0, 29: 0, 30: 0,
                              31: 0, 32: 0, 33: 0, 34: 0, 35: 0, 36: 0,
                              37: 0, 38: 0, 39: 0, 40: 0, 41: 0, 42: 0, 43: 5, 44: 0, 45: 0, 46: 0, 47: 0, 48: 0, 49: 0,
                              50: 0, 51: 0, 52: 0, 53: 0, 54: 0,
                              55: 0, 56: 0, 57: 0, 58: 0, 59: 0, 60: 0, 61: 9, 62: 18, 63: 16, 64: 0, 65: 0, 66: 11,
                              67: 0, 68: 0, 69: 0, 70: 0, 71: 20, 72: 0,
                              73: 0, 74: 0, 75: 0, 76: 0, 77: 0, 78: 0, 79: 0, 80: 0, 81: 0, 82: 0, 83: 0, 84: 0, 85: 0,
                              86: 0, 87: 0, 88: 0, 89: 0, 90: 0, 91: 0,
                              92: 0, 93: 0, 94: 0, 95: 0, 96: 0, 97: 0, 98: 0, 99: 0, 100: 0, 101: 0, 102: 0, 103: 0,
                              104: 0, 105: 0, 106: 0, 107: 0, 108: 0, 109: 0,
                              110: 0, 111: 0, 112: 0, 113: 0, 114: 0, 115: 0, 116: 0, 117: 0, 118: 0, 119: 0, 120: 0,
                              121: 0, 122: 0, 123: 0, 124: 0, 125: 0, 126: 0, 127: 0,
                              128: 0, 129: 0, 130: 0, 131: 0, 132: 0, 133: 0, 134: 0, 135: 0, 136: 0, 137: 0, 138: 0,
                              139: 0, 140: 0, 141: 0, 142: 0, 143: 0, 144: 0, 145: 0,
                              146: 0, 147: 0, 148: 0, 149: 0, 150: 0, 151: 0, 152: 0, 153: 0, 154: 0, 155: 0, 156: 0,
                              157: 0, 158: 0, 159: 0, 160: 0, 161: 0, 162: 0, 163: 0,
                              164: 0, 165: 0, 166: 0, 167: 0, 168: 0, 169: 0, 170: 0, 171: 0, 172: 0, 173: 0, 174: 0,
                              175: 0, 176: 0, 177: 0, 178: 0, 179: 0, 180: 0, 181: 0, }
    def label_transform(self, label):
        temp = label.copy()
        for k, v in self._coco2voc_map.items():
                label[temp == k] = v
        return label

    def __getitem__(self,index):
        item = self.files[index]
        name = item["name"]
        image = cv2.imread(os.path.join(self.root, 'coco/', item["img"]),cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if 'test' in self.list_path and 'val' not in self.list_path:
            size = image.shape
            if self.transform !=None:
                image = self.transform(image) #ToTensor
            return image, np.array(size), name
        else:
            label = cv2.imread(os.path.join(self.root,
                                            'coco',item["label"]),cv2.IMREAD_GRAYSCALE)
            if self.num_classes==21:
                label = self.label_transform(label)
            size = label.shape
            if self.transform!=None:
                image,label = self.transform(image,label) # aug & normlize
            return image,label,np.array(size),name
