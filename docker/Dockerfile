FROM nvidia/cuda:9.0-cudnn7-devel-centos7

WORKDIR /root
COPY dependencies /root/dependencies

############## add these lines when using nvidia-docker ##############
RUN yum-config-manager --disable base/7/x86_64e
RUN yum-config-manager --disable extras/7/x86_64
RUN yum-config-manager --disable cuda
RUN echo "nameserver 8.8.8.8" >> /etc/resolv.conf
RUN yum update -y
#######################################################################


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
#######################################################################


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
#######################################################################


WORKDIR /root/dependencies
RUN bash Anaconda3-2019.03-Linux-x86_64.sh -b
ENV PATH /root/anaconda3/bin:$PATH

##################### pip dependencies #####################
RUN pip install requests

##################### use anaconda Create a Python 3.6 environment #####################
RUN /root/anaconda3/bin/conda install -y conda-build \
 && /root/anaconda3/bin/conda create -y --name HRNet python=3.6 \
 && /root/anaconda3/bin/conda clean -ya

##################### use miniconda Create a Python 3.6 environment #####################
ENV CONDA_DEFAULT_ENV=HRNet
ENV CONDA_PREFIX=/root/anaconda3/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

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

########################### about Coding habits ###########################
RUN echo "set mouse=a" >> ~/.vimrc
RUN echo "set ttymouse=xterm2" >> ~/.vimrc
RUN echo "set paste" >> ~/.vimrc
RUN echo "export PS1='[\u@\h \W]\$ '" >> ~/.bashrc
RUN echo "source activate HRNet" >> ~/.bashrc
RUN echo "alias ns=nvidia-smi" >> ~/.bashrc
RUN echo "alias lsdir='for a in \`find .  -mindepth 1 -maxdepth 1 -type d|sort -r\`; do echo \"\$a:\`find \$a |wc -l\`\"; done'" >> ~/.bashrc
###########################################################################


WORKDIR /home/root
RUN echo "done"
