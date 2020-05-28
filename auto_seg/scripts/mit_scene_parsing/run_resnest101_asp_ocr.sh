nvidia-smi

cd ../../

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 tools/train.py \
                                                                --cfg experiments/mit_scene_parsing/seg_resnest101_trainval_asp_ocr_480x480_sgd_lr1e-3_wd5e-4_bs_20_epoch150.yaml

CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/test.py --cfg experiments/mit_scene_parsing/seg_resnest101_trainval_asp_ocr_480x480_sgd_lr1e-3_wd5e-4_bs_20_epoch150.yaml \
                     DATASET.TEST_SET list/mit_scene_parsing/test.txt \
                     TEST.MODEL_FILE output/mit_scene_parsing/seg_resnest101_trainval_asp_ocr_480x480_sgd_lr1e-3_wd5e-4_bs_20_epoch150/best.pth \
                     TEST.SCALE_LIST 0.5,0.75,1.0,1.25,1.5,1.75 \
                     TEST.FLIP_TEST True
