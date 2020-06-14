import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from tqdm import tqdm
import csv
import cv2
import argparse


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--submit_file", required=True)
    parser.add_argument("--area_thres",required=True)
    parser.add_argument("--min_contour_area",required=True)
    args = parser.parse_args()

    return args


# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction

def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def remove_smallest(mask, min_contour_area):
    _,contours, _= cv2.findContours(
        mask.copy(), cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )
    #print(contours)
    contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]

    background = np.zeros(mask.shape, np.uint8)
    choosen = cv2.drawContours(
        background, contours,
        -1, (255), thickness=cv2.FILLED
    )
    return choosen

def apply_thresholds(mask, area_threshold, min_contour_area):

    if mask.sum() < area_threshold:
        return ""
    choosen =mask
    if min_contour_area>0:
        choosen = remove_smallest(mask,min_contour_area)
    if mask.shape[0] == 768:
        reshaped_mask = choosen
    else:
        reshaped_mask = cv2.resize(
            choosen,
            dsize=(768, 768),
            interpolation=cv2.INTER_LINEAR
        )
    reshaped_mask=(reshaped_mask>0).astype(int)*255
    #print(np.unique(reshaped_mask))
    return rle_encode(reshaped_mask)

def main():
    args = argparser()
    test_path = args.output_dir
    test_list = os.listdir(test_path)
    with open(args.submit_file, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["ImageId", "EncodedPixels"])
        for img in tqdm(test_list):
            name = img.split(".")[0]
            mask = cv2.imread(os.path.join(test_path, img), cv2.IMREAD_GRAYSCALE)
            rle = apply_thresholds(mask, args.area_thres, args.min_contour_area)
            writer.writerow([name + ".jpg", rle])

if __name__=='__main__':
    main()
