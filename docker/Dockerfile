FROM nvidia/cuda:9.0-cudnn7-devel-centos7

ARG TARGET_DIR=/home/root
ARG PYTHON_VERSION=3.6

WORKDIR /root

############## add these lines when using nvidia-docker ##############
RUN yum-config-manager --disable base/7/x86_64e
RUN yum-config-manager --disable extras/7/x86_64
RUN yum-config-manager --disable cuda
RUN echo "nameserver 8.8.8.8" >> /etc/resolv.conf
RUN yum update -y

###################### basic dependencis ##############################
RUN yum install -y \
                bzip2 \
                gcc \
                automake \
                autoconf \
                libtool \
                make \
                gcc-c++ \
                git \
                vim* \
                wget \
                net-tools \
                zip

############################## install miniconda ################################
RUN wget  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /root/miniconda.sh && \
     chmod +x /root/miniconda.sh && \
     /root/miniconda.sh -b -p /opt/conda && \
     /opt/conda/bin/conda install -y python=${PYTHON_VERSION} cython && \
     rm /root/miniconda.sh && \
     /opt/conda/bin/conda clean -ya

ENV PATH /opt/conda/bin:$PATH

RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
RUN conda config --set show_channel_urls yes
RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/

###################### update gcc to gcc-4.9.0 #####################
#WORKDIR /root/source
#RUN wget http://ftp.gnu.org/gnu/gcc/gcc-4.9.0/gcc-4.9.0.tar.bz2
#RUN tar -jxvf gcc-4.9.0.tar.bz2
#WORKDIR gcc-4.9.0
#RUN ./contrib/download_prerequisites
#RUN mkdir build
#WORKDIR build
#RUN ../configure --enable-checking=release --enable-languages=c,c++ --disable-multilib
#RUN make -j4
#RUN make install
#RUN gcc -v

RUN conda install -c serge-sans-paille gcc_49

RUN mv /usr/bin/gcc /usr/bin/gcc-4.8 &&  ln -s /opt/conda/bin/gcc-4.9 /usr/bin/gcc

RUN mv /usr/bin/g++ /usr/bin/g++-4.8 &&  ln -s /opt/conda/bin/g++-4.9 /usr/bin/g++

RUN gcc -v
#RUN ln -s /root/anaconda3/bin/g++-4.9 /opt/conda/bin/g++
#
###################### use miniconda Create a Python 3.6 environment #####################
RUN conda install -y \
    ipython \
    mkl

RUN conda install pytorch==1.2.0 torchvision=0.3.0 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/

##################### pip dependencies #####################
ARG PIP_REPO="-i https://mirrors.aliyun.com/pypi/simple/"

RUN pip install opencv-python $PIP_REPO
RUN pip install Cython $PIP_REPO
RUN pip install llvmlite $PIP_REPO
RUN pip install numba $PIP_REPO
RUN pip install progress $PIP_REPO
RUN pip install matplotlib $PIP_REPO
RUN pip install easydict==1.7 $PIP_REPO
RUN pip install scipy $PIP_REPO
RUN pip install requests $PIP_REPO
RUN pip install Flask $PIP_REPO
RUN pip install gevent $PIP_REPO

######## for HRNet ########
RUN pip install shapely==1.6.4 $PIP_REPO
RUN pip install pandas $PIP_REPO
RUN pip install pyyaml $PIP_REPO
RUN pip install json_tricks $PIP_REPO
RUN pip install scikit-image $PIP_REPO
RUN pip install yacs>=0.1.5 $PIP_REPO
RUN pip install tensorboardX>=1.6 $PIP_REPO
RUN pip install tqdm $PIP_REPO
RUN pip install ninja $PIP_REPO

######## 拷贝并编译代码 ########
COPY auto_seg ${TARGET_DIR}/auto_seg
COPY setup_in_docker.py ${TARGET_DIR}/setup.py
COPY remove_all_c_py_files.sh ${TARGET_DIR}

WORKDIR ${TARGET_DIR}
RUN rm -fr ${TARGET_DIR}/auto_seg/experiments
RUN python setup.py build_ext --inplace
RUN sh remove_all_c_py_files.sh
RUN rm -fr build setup.py remove_all_c_py_files.sh

WORKDIR ${TARGET_DIR}/auto_seg

RUN echo "done"
