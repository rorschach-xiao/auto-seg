#!/usr/bin/env bash
#本脚本用来编译run_docker.py文件， 不对外部提供

python ./setup.py build_ext --inplace

rm -f run_docker.c run_docker.py setup.py
mv run_docker.* run_docker.so

rm -fr build build_python.sh
