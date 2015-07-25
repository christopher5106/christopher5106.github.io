---
layout: post
title:  "Deep learning with Cuda 7, CuDNN 2 and Caffe for Digits 2 and Python on Ubuntu 14.04"
date:   2015-07-16 23:00:51
categories: big data
---

#Install on iMac 27", OS X 10.10.4, NVIDIA GeForce GT 755M 1024 Mo

1\. Install Cuda 7

Check your version

{% highlight bash %}
/usr/local/cuda/bin/nvcc --version
# Cuda compilation tools, release 7.0, V7.0.27
{% endhighlight %}

2\. Download CuDNN

    wget https://s3-eu-west-1.amazonaws.com/christopherbourez/public/cudnn-6.5-osx-v2.tgz
    tar xvzf cudnn-6.5-osx-v2.tgz
    rm cudnn-6.5-osx-v2.tgz
    cd cudnn-6.5-osx-v2/
    sudo cp cudnn.h /usr/local/cuda/include/
    sudo cp libcudnn* /usr/local/cuda/lib/

3\. Install the packages

    brew tap homebrew/science
    brew update
    brew install snappy
    brew install lmdb
    brew install hdf5
    brew install leveldb
    brew install openblas
    brew install glog
    brew install protobuf
    brew install cmake
    brew install opencv
    brew install opencv3

4\. Install boost 1.57 (Caffe is not compatible with Boost 1.58 as explaned [here](http://itinerantbioinformaticist.blogspot.fr/2015/05/caffe-incompatible-with-boost-1580.html)). For that reason change the `/usr/local/Library/Formula/boost.rb` with the contents of [boost.rb 1.57](https://raw.githubusercontent.com/Homebrew/homebrew/6fd6a9b6b2f56139a44dd689d30b7168ac13effb/Library/Formula/boost.rb) and `/usr/local/Library/Formula/boost-python.rb` with the contents of [boost-python.rb 1.57](https://raw.githubusercontent.com/Homebrew/homebrew/3141234b3473717e87f3958d4916fe0ada0baba9/Library/Formula/boost-python.rb).

    brew install boost
    brew install boost-python

5\. Download and install [Anaconda](http://continuum.io/downloads) which is a very great for managing python packages.

    bash Anaconda-2.3.0-MacOSX-x86_64.sh

Install the python packages

    conda install python
    conda install numpy
    conda install hdf5

You can verify the path :

{% highlight bash %}
which python
#/Users/christopherbourez/anaconda/bin/python
{% endhighlight %}


6\. Clone the caffe repository

    git clone https://github.com/BVLC/caffe.git
    cd caffe
    cp Makefile.config.example

7\. Create the configuration file `Makefile.config`

{% highlight makefile %}
CPU_ONLY := 1
CUDA_DIR := /usr/local/cuda
CUDA_ARCH := -gencode arch=compute_20,code=sm_20 \
                -gencode arch=compute_20,code=sm_21 \
                -gencode arch=compute_30,code=sm_30 \
                -gencode arch=compute_35,code=sm_35 \
                -gencode arch=compute_50,code=sm_50 \
                -gencode arch=compute_50,code=compute_50
BLAS := atlas
INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include
LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib
INCLUDE_DIRS += $(shell brew --prefix)/include
LIBRARY_DIRS += $(shell brew --prefix)/lib
USE_PKG_CONFIG := 1

BUILD_DIR := build
DISTRIBUTE_DIR := distribute
TEST_GPUID := 0
Q ?= @
{% endhighlight %}

8\. Build

    mkdir build
    cd build
    make all --jobs=4
    make test
    make runtest
    #make pycaffe
    export CAFFE_HOME=~/caffe


 9\. Download DIGITS

{% highlight bash %}
git clone https://github.com/NVIDIA/DIGITS.git digits-2.0
cd digits-2.0/digits/
sudo pip install -r requirements.txt
./digits-devserver
{% endhighlight %}

and choose `~/caffe` as Caffe path.

Open Port 5000 on the instance. The server will be running at [http://0.0.0.0:5000/](http://0.0.0.0:5000/)

**Well done!**

Troubleshooting :

I had some errors :

Undefined symbols for architecture x86_64:   "cv::imread

=> I cleaned all the opencv

    sudo make uninstall #in the opencv build repo (if you installed from source)
    brew uninstall opencv
    brew uninstall opencv3
    rm /usr/local/lib/libopencv_*
    sudo rm -r /usr/local/share/OpenCV
    sudo rm -r /usr/local/include/opencv2

Be careful to brew messages, all libraries have to be linked.

Undefined symbols for architecture x86_64:
  "google::SetUsageMessage

=> I re-installed gflags

Undefined symbols for architecture x86_64:
"leveldb::

=> I re-installed leveldb

Undefined symbols for architecture x86_64: "google::protobuf:

=> I re-installed protobuf.
