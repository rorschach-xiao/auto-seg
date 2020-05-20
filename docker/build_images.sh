#!/usr/bin/env bash
#本脚本用来编译Docker镜像，不对外部提供

IMAGE_VERSION=v0.1

cp -r ../autocv_classification_pytorch ./

# 下载pretrained models
if ! ls dependencies/models;then
    mkdir -p dependencies/models

    model_list="hrnetv2_w18_imagenet_pretrained.pth \
                hrnetv2_w32_imagenet_pretrained.pth  \
                hrnetv2_w48_imagenet_pretrained.pth  \
                resnest101-22405ba7.pth  \
                resnest200-75117900.pth  \
                resnest50-528c19ca.pth"
    for model in ${model_list[@]};do
        wget ftp://m7-model-gpu16.4pd.io/autocv_seg/models/${model} -P ./dependencies/models
    done
fi

docker build -t autocv/classifier:${IMAGE_VERSION} .
