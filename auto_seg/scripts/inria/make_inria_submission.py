import numpy as np
import cv2
import os
from tqdm import tqdm
from scipy import misc
from PIL import Image
import argparse

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir",required=True)
    parser.add_argument("--submit_dir",required=True)
    args = parser.parse_args()

    return args

def main():
    args = argparser()
    test_results_path = args.output_dir
    test_results_list = os.listdir(test_results_path)
    convert_results_path = args.submit_dir
    large_img_list = []
    if not os.path.exists(convert_results_path):
        os.makedirs(convert_results_path)

    for i in test_results_list:
        origin_img_name = i.split("_")[0]
        if (origin_img_name not in large_img_list):
            large_img_list.append(origin_img_name)

    for img in tqdm(large_img_list):
        large_img = np.zeros((5000, 5000)).astype(np.uint8)
        for i in range(1, 101):
            img_full_path = os.path.join(test_results_path, img + "_{}.png".format(i))
            if (os.path.exists(img_full_path)):
                image = cv2.imread(img_full_path, cv2.IMREAD_GRAYSCALE)
                image[image == 1] = 255
                row = int((i - 1) / 10) * 500
                col = ((i - 1) % 10) * 500
                # print(row,col)
                large_img[row:row + 500, col:col + 500] += image
        Image.fromarray(large_img).save(os.path.join(convert_results_path, img + ".tif"))

if __name__ =='__main__':
    main()



