#!/usr/bin/env bash

cd auto-seg
files=`ls`
for f in ${files[@]}; do
    if [[ -d $f ]];then
         find ./$f/ -name "*.py" | xargs rm -fr
         find ./$f/ -name "*.c" | xargs rm -fr
    fi
done
#find ./ -name "*.py" | xargs rm -fr
#find ./ -name "*.c" | xargs rm -fr
rm -fr autocv_cls_op.*
