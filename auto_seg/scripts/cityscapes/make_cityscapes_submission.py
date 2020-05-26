import os

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse

ignore_label = 255
label_mapping = {-1: ignore_label, 0: ignore_label,
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

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir",required=True)
    parser.add_argument("--submit_dir",required=True)
    args = parser.parse_args()

    return args
def convert_label(label, inverse=False):
    temp = label.copy()
    if inverse:
        for v, k in label_mapping.items():
            label[temp == k] = v
    else:
        for k, v in label_mapping.items():
            label[temp == k] = v
    return label

def main():
    args = argparser()
    result_path = args.pred_dir
    result_list = os.listdir(result_path)
    os.makedirs(args.submit_dir, exist_ok=True)

    for label in tqdm(result_list):
        label_img = cv2.imread(os.path.join(result_path,label),cv2.IMREAD_GRAYSCALE)
        label_convert = convert_label(label_img,inverse=True)
        save_img = Image.fromarray(label_convert)
        save_img.save(os.path.join(args.submit_dir,label))

if __name__=="__main__":
    main()
