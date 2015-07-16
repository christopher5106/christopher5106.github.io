---
layout: post
title:  "Deep learning with Cuda, CuDNN and Caffe for Digits and Python on Mac OS X"
date:   2015-07-16 23:00:51
categories: big data
---


1. Install CUDA

2. Download CuDNN

    tar xvzf cudnn-6.5-osx-v2.tgz
    rm cudnn-6.5-osx-v2.tgz
    cd cudnn-6.5-osx-v2/
    sudo cp cudnn.h /usr/local/cuda/include/

3. Install the package

    brew install opencv
    brew install boost
    brew install snappy
    brew install lmdb
    brew install hdf5
    brew install leveldb
    brew install openblas
    brew install glog
    brew install protobuf


4. Install the python packages

    conda install numpy
    conda install hdf5


5. Clone the caffe repository

    git clone https://github.com/BVLC/caffe.git
    cd caffe
    cp Makefile.config.example Makefile.config

and edit the configuration

    USE_CUDNN := 1

    BLAS := open

    BLAS_INCLUDE := $(shell brew --prefix openblas)/include
    BLAS_LIB := $(shell brew --prefix openblas)/lib

    ANACONDA_HOME := $(HOME)/anaconda
    PYTHON_INCLUDE := $(ANACONDA_HOME)/include \
             $(ANACONDA_HOME)/include/python2.7 \
             $(ANACONDA_HOME)/lib/python2.7/site-packages/numpy/core/include \

    WITH_PYTHON_LAYER := 1

and build

    make all


6. Download DIGITS

    tar xvzf digits-2.0.0-preview.gz
    cd digits-2.0
