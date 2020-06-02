nvidia-smi

cd ../../

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 tools/train.py --cfg experiments/LIP/seg_resnest101_trainval_asp_ocr_473x473_sgd_lr1e-3_wd5e-4_bs_16_epoch120.yaml

CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/test.py --cfg experiments/LIP/seg_resnest101_trainval_asp_ocr_473x473_sgd_lr1e-3_wd5e-4_bs_16_epoch120.yaml \
                     DATASET.TEST_SET list/lip/testList.txt \
                     TEST.MODEL_FILE output/lip/seg_resnest101_trainval_asp_ocr_473x473_sgd_lr1e-3_wd5e-4_bs_16_epoch120/best.pth \
                     TEST.FLIP_TEST True \
                     TEST.SCALE_LIST 0.5,0.75,1.0,1.25,1.5,1.75
