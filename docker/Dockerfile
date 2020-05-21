FROM nvidia/cuda:9.0-cudnn7-devel-centos7

ARG TARGET_DIR=/home/root
ARG PYTHON_VERSION=3.6

WORKDIR /root

############## add these lines when using nvidia-docker ##############
#RUN yum-config-manager --disable base/7/x86_64e
#RUN yum-config-manager --disable extras/7/x86_64
#RUN yum-config-manager --disable cuda
#RUN echo "nameserver 8.8.8.8" >> /etc/resolv.conf
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

###################### update gcc to gcc-4.9.0 #####################
WORKDIR /root/source
RUN wget http://ftp.gnu.org/gnu/gcc/gcc-4.9.0/gcc-4.9.0.tar.bz2
RUN tar -jxvf gcc-4.9.0.tar.bz2
WORKDIR gcc-4.9.0
RUN ./contrib/download_prerequisites
RUN mkdir build
WORKDIR build
RUN ../configure --enable-checking=release --enable-languages=c,c++ --disable-multilib
RUN make -j4
RUN make install
RUN gcc -v

############################## install miniconda #################################
RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     /opt/conda/bin/conda install -y python=${PYTHON_VERSION} cython && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda clean -ya

ENV PATH /opt/conda/bin:$PATH

##################### use miniconda Create a Python 3.6 environment #####################
RUN conda install -y \
    ipython \
    mkl

RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
RUN conda config --set show_channel_urls yes
RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
RUN conda install pytorch==1.2.0 torchvision=0.3.0 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/

##################### pip dependencies #####################
RUN pip install opencv-python
RUN pip install Cython
RUN pip install llvmlite
RUN pip install numba
RUN pip install progress
RUN pip install matplotlib
RUN pip install easydict==1.7
RUN pip install scipy
RUN pip install requests
RUN pip install Flask
RUN pip install gevent

######## for HRNet ########
RUN pip install shapely==1.6.4
RUN pip install pandas
RUN pip install pyyaml
RUN pip install json_tricks
RUN pip install scikit-image
RUN pip install yacs>=0.1.5
RUN pip install tensorboardX>=1.6
RUN pip install tqdm
RUN pip install ninja

######## 拷贝并编译代码 ########
COPY auto-seg ${TARGET_DIR}
COPY setup_in_docker.py ${TARGET_DIR}
COPY remove_all_c_py_files.sh ${TARGET_DIR}

WORKDIR ${TARGET_DIR}/auto-seg

RUN rm -fr ${TARGET_DIR}/auto-seg/experiments
RUN python setup.py build_ext --inplace
RUN sh remove_all_c_py_files.sh
RUN rm -fr build setup.py remove_all_c_py_files.sh

RUN echo "done"
