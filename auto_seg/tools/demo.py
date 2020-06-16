import _init_paths
import torch
from tools.auto_run import *
from utils.utils import check_data_format
import cv2
from PIL import Image

if __name__ =='__main__':
    #check_data_format('data/Inria', is_training_data=True)
    #train('data/optic','output','0,1')
    #test('data/optic','output/custom_2020-06-16-06-44','0,1')
    inferer = InferenceJob('output/custom_2020-06-16-06-44', cuda_visible_devices='0,1')
    inferer._run_video('data/optic/demo.avi')
    #inferer.frame2video('data/optic/frames_out_dir','demo.avi','optic_pred.avi',10,1,38,(512,512),'MJPG')
    # with open("data/optic/frames_dir/demo_1.jpg",'rb') as f:
    # with open("data/optic/JPEGImages/N0150.jpg", 'rb') as f:
    #     raw_img = f.read()
    #     img = inferer._run(raw_img)
    #
    #     save_img = Image.fromarray(img)
    #     save_img.save(os.path.join("output/custom_2020-06-16-06-44/results.png"))
       #cv2.imwrite("output/custom_2020-06-16-06-44/results.png",img)

    #check_data_format('data/APSIS','testval.txt')

