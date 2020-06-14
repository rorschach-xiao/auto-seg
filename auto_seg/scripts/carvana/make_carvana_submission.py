import numpy as np
import cv2
from PIL import Image
import os
import csv
from tqdm import tqdm
import argparse

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--submit_file", required=True)
    args = parser.parse_args()

    return args

def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

def main():
    args = argparser()
    masks_path = args.output_dir
    submit_file = args.submit_file
    mask_list = os.listdir(masks_path)
    with open(os.path.join(submit_file),"w") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["img","rle_mask"])
        for mask in tqdm(mask_list):
            name=mask.split(".")[0]
            #img = Image.open(os.path.join(masks_path,mask))
            img = cv2.imread(os.path.join(masks_path,mask),cv2.IMREAD_GRAYSCALE)
            #print(img.shape)
            rle = rle_encode(img)
            csv_writer.writerow([name+".jpg",rle])

if __name__=='__main__':
    main()
