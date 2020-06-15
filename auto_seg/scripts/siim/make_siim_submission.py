from tqdm import tqdm
import cv2
from tqdm import tqdm
import numpy as np
import csv
from mask_functions import mask2rle
import os
import argparse

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--submit_file", required=True)
    parser.add_argument("--area_thres",required=True)
    parser.add_argument("--min_contour_area",required=True)
    args = parser.parse_args()

    return args

def apply_thresholds(mask, area_threshold, min_contour_area):
    if mask.sum() < area_threshold:
        return -1

    if min_contour_area > 0:
        choosen = remove_smallest(mask, min_contour_area)
    if mask.shape[0] == 1024:
        reshaped_mask = choosen
    else:
        reshaped_mask = cv2.resize(
            choosen,
            dsize=(1024, 1024),
            interpolation=cv2.INTER_LINEAR
        )
    reshaped_mask = (reshaped_mask > 0).astype(int) * 255
    # print(np.unique(reshaped_mask))
    return mask2rle(reshaped_mask.T, 1024, 1024)


def remove_smallest(mask, min_contour_area):
    contours, _ = cv2.findContours(
        mask.copy(), cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )
    # print(contours)
    contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]

    background = np.zeros(mask.shape, np.uint8)
    choosen = cv2.drawContours(
        background, contours,
        -1, (255), thickness=cv2.FILLED
    )
    return choosen


def main():
    args=argparser()
    test_path = args.output_dir
    test_list = os.listdir(test_path)
    rle_dict = {}
    with open(args.submit_file, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["ImageId", "EncodedPixels"])
        for img in tqdm(test_list):
            name = img.split(".")[0]
            mask = cv2.imread(os.path.join(test_path, img), cv2.IMREAD_GRAYSCALE)
            rle = apply_thresholds(mask, int(args.area_thres), int(args.min_contour_area))
            if rle == "":
                # print("empty mask!")
                rle = "-1"
            writer.writerow([name, rle])

if __name__ == '__main__':
    main()