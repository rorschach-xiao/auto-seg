import _init_paths
from tools.auto_run import *
from utils.utils import check_data_format
import cv2

if __name__ =='__main__':
    #train('data/optic','output','2,3')
    #test('data/optic','output/custom_2020-05-22-09-44','2,3')
    #inferer = InferenceJob('output/custom_2015-11-19-18-08', cuda_visible_devices='0,1')
    #with open("data/optic/JPEGImages/P0176.jpg",'rb') as f:
    #    raw_img = f.read()
    #    img = inferer._run(raw_img)
    #    cv2.imwrite(img,"output/custom_2015-11-19-18-08/results.png")

    check_data_format('data/optic','testval.txt')
