nvidia-smi

cd ../../

python -m torch.distributed.launch --nproc_per_node=4 tools/train.py --cfg experiments/voc_sbd/seg_resnest101_asp_ocr_trainval_cls20_473x473_sgd_lr1e-3_wd1e-4_bs_16_epoch80.yaml

python tools/test.py --cfg experiments/voc_sbd/seg_resnest101_asp_ocr_trainval_cls20_473x473_sgd_lr1e-3_wd1e-4_bs_16_epoch80.yaml \
                     DATASET.TEST_SET list/voc_sbd/test2012.txt \
                     TEST.MODEL_FILE output/voc_sbd/seg_resnest101_asp_ocr_trainval_cls20_473x473_sgd_lr1e-3_wd1e-4_bs_16_epoch80/best.pth \
                     TEST.SCALE_LIST 0.5,0.75,1.0,1.25,1.5,1.75 \
                     TEST.FLIP_TEST True
