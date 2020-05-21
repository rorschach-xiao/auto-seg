#!/usr/bin/env bash
#本脚本用来编译Docker镜像，不对外部提供

IMAGE_VERSION=v0.1

REPO_NAME=auto_seg
MODEL_FOLDER=$REPO_NAME/pretrained_models

rm -fr ./${REPO_NAME}
cp -r ../${REPO_NAME} ./

# 下载pretrained models
if ! ls ${MODEL_FOLDER};then
    mkdir -p ${MODEL_FOLDER}

    model_list="hrnetv2_w18_imagenet_pretrained.pth \
                hrnetv2_w32_imagenet_pretrained.pth  \
                hrnetv2_w48_imagenet_pretrained.pth  \
                resnest101-22405ba7.pth  \
                resnest200-75117900.pth  \
                resnest50-528c19ca.pth"
    for model in ${model_list[@]};do
        wget ftp://m7-model-gpu16.4pd.io/autocv_seg/models/${model} -P ${MODEL_FOLDER}
    done
fi

docker build -t autocv/segment:${IMAGE_VERSION} .
