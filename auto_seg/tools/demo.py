import _init_paths
from tools.auto_run import *
from utils.utils import check_data_format
import cv2

if __name__ =='__main__':
    #check_data_format('data/Inria', is_training_data=True)
    #train('data/Inria','output','2,3')
    #test('data/Inria','output/custom_2020-05-24-18-17','2,3')
    inferer = InferenceJob('output/custom_2020-05-24-18-17', cuda_visible_devices='0,1')
    #inferer._run_video('data/APSIS/demo.avi')
    inferer.frame2video('data/APSIS/frames_out_dir','demo.avi','apsis_pred.avi',10,1,314,(600,800),'MJPG')
    #with open("data/optic/JPEGImages/P0176.jpg",'rb') as f:
    #    raw_img = f.read()
    #    img = inferer._run(raw_img)
    #    cv2.imwrite(img,"output/custom_2015-11-19-18-08/results.png")

    #check_data_format('data/APSIS','testval.txt')

