nvidia-smi

cd ../../
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 tools/train.py \
                                                                --cfg experiments/cityscapes/seg_resnest101_asp_ocr_trainval_768x768_sgd_lr1e-3_wd5e-4_bs_8_epoch240.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/test.py --cfg experiments/cityscapes/seg_resnest101_asp_ocr_trainval_768x768_sgd_lr1e-3_wd5e-4_bs_8_epoch240.yaml \
                     DATASET.TEST_SET list/cityscapes/test.lst \
                     TEST.MODEL_FILE output/cityscapes/seg_resnest101_asp_ocr_trainval_768x768_sgd_lr1e-3_wd5e-4_bs_8_epoch240/best.pth \
                     TEST.SCALE_LIST 0.5,0.75,1.0,1.25,1.5,1.75 \
                     TEST.FLIP_TEST True

python scripts/cityscapes/make_cityscapes_submission.py --pred_dir output/cityscapes/seg_resnest101_asp_ocr_trainval_768x768_sgd_lr1e-3_wd5e-4_bs_8_epoch240/test_resutls \
                                                        --submit_dir output/cityscapes/seg_resnest101_asp_ocr_trainval_768x768_sgd_lr1e-3_wd5e-4_bs_8_epoch240/sub_resutls

python scripts/cityscapes/segfix.py --input  output/cityscapes/seg_resnest101_asp_ocr_trainval_768x768_sgd_lr1e-3_wd5e-4_bs_8_epoch240/sub_resutls \
                                    --offset data/cityscapes/offset_semantic/test_offset/semantic/offset_hrnext/ \
                                    --split  test
                                    --out  output/cityscapes/seg_resnest101_asp_ocr_trainval_768x768_sgd_lr1e-3_wd5e-4_bs_8_epoch240/segfix_resutls



