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
        sudo cp lib* /usr/local/cuda/lib/

3. Install the package

        brew update
        brew install opencv
        brew install boost
        brew install snappy
        brew install lmdb
        brew install hdf5
        brew install leveldb
        brew install openblas
        brew install glog
        brew install protobuf
        brew install cmake
        brew install boost-python


4. Install the python packages

        conda install numpy
        conda install hdf5
        conda install boost


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

    #PYTHON_LIB := /usr/lib
    PYTHON_LIB := $(ANACONDA_HOME)/lib

    PYTHON_INCLUDE += $(dir $(shell python -c 'import numpy.core; print(numpy.core.__file__)'))/include
    PYTHON_LIB += $(shell brew --prefix numpy)/lib

    WITH_PYTHON_LAYER := 1




and build

    mkdir build
    cd build
    cmake ..
    make all
    make test
    make runtest
    make pycaffe


6. Download DIGITS

        tar xvzf digits-2.0.0-preview.gz
        cd digits-2.0
