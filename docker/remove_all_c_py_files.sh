#!/usr/bin/env bash

cd auto_seg
files=`ls`
for f in ${files[@]}; do
    if [[ -d $f ]];then
         find ./$f/ -name "*.py" | grep -v "start_server.py" | grep -v "auto_run.py" \
             |grep -v "auto_helper.py" | xargs rm -fr
         find ./$f/ -name "*.c" | xargs rm -fr
    fi
done
#find ./ -name "*.py" | xargs rm -fr
#find ./ -name "*.c" | xargs rm -fr
rm -fr build
