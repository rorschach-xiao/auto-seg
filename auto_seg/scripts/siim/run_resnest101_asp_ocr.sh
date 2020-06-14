nvidia-smi
cd ../../

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 tools/train.py \
                                                                --cfg experiments/siim/seg_resnest101_asp_ocr_512x512_bce_sgd_lr7e-3_wd5e-4_bs_12_epoch80.yaml

CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/test.py --cfg experiments/siim/seg_resnest101_asp_ocr_512x512_bce_sgd_lr7e-3_wd5e-4_bs_12_epoch80.yaml \
                     DATASET.TEST_SET list/siim/test.txt \
                     TEST.MODEL_FILE output/siim/seg_resnest101_asp_ocr_512x512_bce_sgd_lr7e-3_wd5e-4_bs_12_epoch80/final_state.pth \
                     TEST.SCALE_LIST 0.5,0.75,1.0,1.25,1.5 \
                     TEST.FLIP_TEST True

python scripts/siim/make_siim_submission.py --output_dir output/siim/seg_resnest101_asp_ocr_512x512_bce_sgd_lr7e-3_wd5e-4_bs_12_epoch80/test_results \
                                            --submit_file output/siim/seg_resnest101_asp_ocr_512x512_bce_sgd_lr7e-3_wd5e-4_bs_12_epoch80/stage_2_submission_resnest101_asp_ocr.csv \
                                            --area_thres 1000 \
                                            --min_contour_area 100

