nvidia-smi
cd ../../

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 tools/train.py  \
                                                                                   --cfg experiments/carvana/seg_resnest101_asp_ocr_bce_640x960_sgd_lr7e-3_wd5e-4_bs_8_epoch80.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/test.py --cfg experiments/carvana/seg_resnest101_asp_ocr_bce_640x960_sgd_lr7e-3_wd5e-4_bs_8_epoch80.yaml \
                     DATASET.TEST_SET list/carvana/test_hq.txt \
                     TEST.MODEL_FILE output/carvana/seg_resnest101_asp_ocr_bce_640x960_sgd_lr7e-3_wd5e-4_bs_8_epoch80/best.pth \
                     TEST.SCALE_LIST 0.5,0.75,1.0,1.25,1.5 \
                     TEST.FLIP_TEST True
python scripts/carvana/make_carvana_submission.py --output_dir output/carvana/seg_resnest101_asp_ocr_bce_640x960_sgd_lr7e-3_wd5e-4_bs_8_epoch80/test_results \
                                                  --submit_file output/carvana/seg_resnest101_asp_ocr_bce_640x960_sgd_lr7e-3_wd5e-4_bs_8_epoch80/resnest101_asp_ocr_submission.csv

zip  output/carvana/seg_resnest101_asp_ocr_bce_640x960_sgd_lr7e-3_wd5e-4_bs_8_epoch80/resnest101_asp_ocr_submission.csv.zip \
     output/carvana/seg_resnest101_asp_ocr_bce_640x960_sgd_lr7e-3_wd5e-4_bs_8_epoch80/resnest101_asp_ocr_submission.csv