import _init_paths
from tools.auto_run import *


if __name__ =='__main__':
    #train('data/optic','output','2,3')
    #test('data/optic','output/custom_2015-11-19-18-08','2,3')
    inferer = InferenceJob('output/custom_2015-11-19-18-08', cuda_visible_devices='0,1')
    with open("data/optic/JPEGImages/P0176.jpg",'rb') as f:
        raw_img = f.read()
        inferer._run(raw_img)

