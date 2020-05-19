nvidia-smi

cd ../../
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_addr='127.0.0.1' --master_port=2333 tools/train.py \
                                                                                                            --cfg experiments/apsis/seg_hrnet_w32_512x512_adam_lr1e-3_wd1e-4_bs_8_epoch80.yaml

CUDA_VISIBLE_DEVICES=4,5,6,7 python tools/test.py --cfg experiments/apsis/seg_hrnet_w32_512x512_adam_lr1e-3_wd1e-4_bs_8_epoch80.yaml \
                     DATASET.TEST_SET list/APSIS/testval.txt \
                     TEST.MODEL_FILE output/apsis/seg_hrnet_w32_512x512_adam_lr1e-3_wd1e-4_bs_8_epoch80/final_state.pth \
                     TEST.FLIP_TEST True

