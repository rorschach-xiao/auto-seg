# auto-seg



- ### 目前最好效果

|         dataset         |  Metric  | score(%) |       Network       |
| :---------------------: | :------: | :------: | :-----------------: |
|       Coco-stuff        |   mIoU   |  41.05   | ResNeSt101+ASPP+OCR |
|       Cityscapes        |   mIoU   |  81.56   |     HRNet48+OCR     |
|     Pascal VOC2012      |   mIoU   |  84.26   | ResNeSt101+ASPP+OCR |
|           LIP           |   mIoU   |  57.55   | ResNeSt101+ASPP+OCR |
|     Pascal Context      |   mIoU   |  57.72   | ResNeSt101+ASPP+OCR |
|    MIT Scene Parsing    |   mIoU   |  37.94   | ResNeSt101+ASPP+OCR |
| bdd100k domain adaption |   mIoU   |  53.81   |     HRNet48+OCR     |
|    bdd100k drivable     |   mIoU   |  85.06   |     HRNet48+OCR     |
|        SIIM-ACR         |   Dice   |  85.20   |     HRNet48+OCR     |
|         airbus          | F1-score |          |                     |
|         Carvana         |   Dice   |  99.631  | ResNeSt101+ASPP+OCR |
|       KolektorSDD       |   Dice   |  93.78   |     HRNet48+OCR     |
|          Inria          |   mIoU   |  76.99   |     HRNet48+OCR     |
|          APSIS          |   Dice   |  98.41   |     HRNet48+OCR     |
|        Magnetic         |   Dice   |  82.06   | ResNeSt101+ASPP+OCR |
|       CrackForest       |   mIoU   |  63.54   |     HRNet48+OCR     |



- ### 训练和测试命令

  - Train

    ```shell
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 tools/train.py --cfg experiments/'your config file' 
    ```

    

  - Test

    ```shell
    python tools/test.py --cfg experiments/'your config file' \
                         DATASET.TEST_SET 'your data list path' \
                         TEST.MODEL_FILE 'your model checkpoint file path' \
                         TEST.SCALE_LIST 'test scale list,default [1]' \
                         TEST.FLIP_TEST True
    ```

    

  

  

​	