nvidia-smi
cd ../../
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr="127.0.0.1" --master_port=6666 tools/train.py --cfg experiments/cityscapes/seg_hrnet_ocr_w48_trainval_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml

CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/test.py --cfg experiments/cityscapes/seg_hrnet_ocr_w48_trainval_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml \
                     DATASET.TEST_SET list/cityscapes/test.lst \
                     TEST.MODEL_FILE output/cityscapes/seg_hrnet_ocr_w48_trainval_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484/best.pth \
                     TEST.SCALE_LIST 0.5,0.75,1.0,1.25,1.5,1.75 \
                     TEST.FLIP_TEST True

python scripts/cityscapes/make_cityscapes_submission.py --pred_dir output/cityscapes/seg_hrnet_ocr_w48_trainval_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484/test_results/ \
                                                        --submit_dir output/cityscapes/seg_hrnet_ocr_w48_trainval_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484/sub_results/

python scripts/cityscapes/segfix.py --input output/cityscapes/seg_hrnet_ocr_w48_trainval_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484/sub_results/ \
                                    --offset data/cityscapes/test_offset/sementic/offset_hrnext \
                                    --out output/cityscapes/seg_hrnet_ocr_w48_trainval_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484/segfix_sub_results

