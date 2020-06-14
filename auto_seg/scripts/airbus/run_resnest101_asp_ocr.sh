nvidia-smi
cd ../../

CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4  tools/train.py \
                                                                                    --cfg experiments/airbus/seg_resnest101_asp_ocr_bce_512x512_sgd_lr7e-3_wd5e-4_bs_12_epoch80.yaml

CUDA_VISIBLE_DEVICES=4,5,6,7 python tools/test.py --cfg experiments/airbus/seg_resnest101_asp_ocr_bce_512x512_sgd_lr7e-3_wd5e-4_bs_12_epoch80.yaml \
                     DATASET.TEST_SET list/airbus/test.txt \
                     TEST.MODEL_FILE output/airbus/seg_resnest101_asp_ocr_bce_512x512_sgd_lr7e-3_wd5e-4_bs_12_epoch80/final_state.pth \
                     TEST.SCALE_LIST 0.5,0.75,1.0,1.25,1.5,1.75,2.0 \
                     TEST.FLIP_TEST True

python scripts/airbus/make_airbus_submission.py --output_dir output/airbus/seg_resnest101_asp_ocr_bce_512x512_sgd_lr7e-3_wd5e-4_bs_12_epoch80/test_results \
                                                --submit_file output/airbus/seg_resnest101_asp_ocr_bce_512x512_sgd_lr7e-3_wd5e-4_bs_12_epoch80/testv2_submission_resnest101_asp_ocr.csv \
                                                --area_thres 200
                                                --min_contour_area 50
