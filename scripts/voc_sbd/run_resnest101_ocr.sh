nvidia-smi
cd ../../
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr="127.0.0.1" --master_port=2323 tools/train.py --cfg experiments/voc_sbd/seg_hrnet_w48_ocr_cls20_520x520_sgd_lr1e-3_wd1e-4_bs_16_epoch80.yaml

CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/test.py --cfg experiments/voc_sbd/seg_resnest101_ocr_cls20_473x473_sgd_lr1e-3_wd1e-4_bs_16_epoch80.yaml\
                     DATASET.TEST_SET list/voc_sbd/test2012.txt \
                     TEST.MODEL_FILE output/voc_sbd/seg_resnest101_ocr_cls20_473x473_sgd_lr1e-3_wd1e-4_bs_16_epoch80/best.pth \
                     TEST.SCALE_LIST 0.5,0.75,1.0,1.25,1.5,1.75 \
                     TEST.FLIP_TEST True
