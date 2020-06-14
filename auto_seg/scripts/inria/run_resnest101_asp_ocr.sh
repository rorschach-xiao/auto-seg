nvidia-smi
cd ../../

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 tools/train.py \
                                                                                   -- cfg experiments/inria/seg_resnest101_asp_ocr_500x500_bce_sgd_lr7e-3_wd5e-4_bs_12_epoch80.yaml

CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/test.py --cfg experiments/inria/seg_resnest101_asp_ocr_500x500_bce_sgd_lr7e-3_wd5e-4_bs_12_epoch80.yaml \
                     DATASET.TEST_SET list/Inria/test.txt \
                     TEST.MODEL_FILE output/inria/seg_resnest101_asp_ocr_500x500_bce_sgd_lr7e-3_wd5e-4_bs_12_epoch80/best.pth \
                     TEST.SCALE_LIST 0.5,0.75,1.0,1.25,1.5 \
                     TEST.FLIP_TEST True

python scripts/inria/make_inria_submission.py --output_dir output/inria/seg_resnest101_asp_ocr_500x500_bce_sgd_lr7e-3_wd5e-4_bs_12_epoch80/test_results \
                                              --submit_dir output/inria/seg_resnest101_asp_ocr_500x500_bce_sgd_lr7e-3_wd5e-4_bs_12_epoch80/submit_results

python scripts/inria/compress.py output/inria/seg_resnest101_asp_ocr_500x500_bce_sgd_lr7e-3_wd5e-4_bs_12_epoch80/submit_results  \
                                 output/inria/seg_resnest101_asp_ocr_500x500_bce_sgd_lr7e-3_wd5e-4_bs_12_epoch80/compress_results

