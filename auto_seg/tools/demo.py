import _init_paths
import torch
from .train import get_sampler
from tools.auto_run import *
from utils.utils import check_data_format
import cv2

if __name__ =='__main__':
    config.TRAIN.TRANS_LIST = ['random_scale',
                               'random_rotate',
                               'random_blur',
                               'random_hflip',
                               'crop',
                               'totensor',
                               'normalize']
    train_transform_list = get_train_transform(config)
    train_transform = Compose(train_transform_list)
    # train data argumentation
    train_dataset = eval('datasets.' + config.DATASET.DATASET)(
        root=config.DATASET.ROOT,
        list_path=config.DATASET.TRAIN_SET,
        num_samples=None,
        num_classes=config.DATASET.NUM_CLASSES,
        transform=train_transform)

    train_sampler = get_sampler(train_dataset)

    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=config.TRAIN.SHUFFLE and train_sampler is None,
        num_workers=config.WORKERS,
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler)

    #check_data_format('data/Inria', is_training_data=True)
    train('data/optic','output','0,1')
    #test('data/APSIS','output/custom_2020-06-01-22-56','2,3')
    #inferer = InferenceJob('output/custom_2020-05-24-18-17', cuda_visible_devices='0,1')
    #inferer._run_video('data/APSIS/demo.avi')
    #inferer.frame2video('data/APSIS/frames_out_dir','demo.avi','apsis_pred.avi',10,1,314,(600,800),'MJPG')
    #with open("data/optic/JPEGImages/P0176.jpg",'rb') as f:
    #    raw_img = f.read()
    #    img = inferer._run(raw_img)
    #    cv2.imwrite(img,"output/custom_2015-11-19-18-08/results.png")

    #check_data_format('data/APSIS','testval.txt')

