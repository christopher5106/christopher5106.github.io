---
layout: post
title:  "Deep learning with Cuda 7, CuDNN 2 and Caffe for Digits 2 and Python on Ubuntu 14.04"
date:   2015-07-16 23:00:51
categories: big data
---

Install on a AWS g2 instance, with Ubuntu 14.04.

#Install Cuda and Cudnn

{% highlight bash %}
#Install Cuda
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_7.0-28_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1404_7.0-28_amd64.deb
sudo apt-get update
sudo apt-get -y install cuda
#check everything ok
/usr/local/cuda/bin/nvcc --version
#> Cuda compilation tools, release 7.0, V7.0.27

#Install Cudnn
wget https://s3-eu-west-1.amazonaws.com/christopherbourez/public/cudnn-6.5-osx-v2.tgz
tar xvzf cudnn-6.5-linux-x64-v2.tgz
cd cudnn-6.5-linux-x64-v2/
sudo cp cudnn.h /usr/local/cuda/include/
sudo cp libcudnn* /usr/local/cuda/lib64/

#Install Git
sudo apt-get install git
{% endhighlight %}

#Install Digits with Digits'Caffe...

{% highlight bash %}
git clone https://github.com/NVIDIA/DIGITS.git digits-2.0
cd digits-2.0/
./install.sh
cd caffe
sudo apt-get install --no-install-recommends libboost-all-dev #missing
make all --jobs=8
cd ../digits/
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib
sudo ln /dev/null /dev/raw1394
sudo pip install -r requirements.txt
./digits-devserver
{% endhighlight %}

and choose `../caffe` as Caffe path.

Open Port 5000 on the instance. The server will be running at [http://0.0.0.0:5000/](http://0.0.0.0:5000/)

Note : the `Makefile`

{% highlight makefile %}
USE_CUDNN := 1
CUDA_DIR := /usr/local/cuda
CUDA_ARCH := -gencode arch=compute_20,code=sm_20 \
                -gencode arch=compute_20,code=sm_21 \
                -gencode arch=compute_30,code=sm_30 \
                -gencode arch=compute_35,code=sm_35 \
                -gencode arch=compute_50,code=sm_50 \
                -gencode arch=compute_50,code=compute_50
BLAS := atlas
PYTHON_INCLUDE := /usr/include/python2.7 \
                /usr/lib/python2.7/dist-packages/numpy/core/include \
                /usr/local/lib/python2.7/dist-packages/numpy/core/include
PYTHON_LIB := /usr/lib
INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include
LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib
BUILD_DIR := build
DISTRIBUTE_DIR := distribute
TEST_GPUID := 0
Q ?= @
LIBRARY_NAME_SUFFIX := -nv
{% endhighlight %}


#... or install your own Caffe with Ananconda

{% highlight bash %}
#Install Anaconda
wget https://3230d63b5fc54e62148e-c95ac804525aac4b6dba79b00b39d1d3.ssl.cf1.rackcdn.com/Anaconda-2.3.0-Linux-x86_64.sh
bash Anaconda-2.3.0-Linux-x86_64.sh
conda install python

#Install Caffe requirements
sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev
sudo apt-get install --no-install-recommends libboost-all-dev
sudo apt-get install libgflags-dev libgoogle-glog-dev liblmdb-dev protobuf-compiler
sudo apt-get install libatlas-base-dev
sudo apt-get remove libopenblas-base
sudo ldconfig /usr/local/cuda/lib64
sudo ln /dev/null /dev/raw1394
conda install protobuf  

#Download Digits
tar xvzf digits-2.0.0-preview.gz
cd digits-2.0/caffe

#Edit the Makefile
vi Makefile.config
{% endhighlight %}

The `Makefile.config` is :

{% highlight makefile %}
USE_CUDNN := 1
CUDA_DIR := /usr/local/cuda
CUDA_ARCH := -gencode arch=compute_20,code=sm_20 \
                -gencode arch=compute_20,code=sm_21 \
                -gencode arch=compute_30,code=sm_30 \
                -gencode arch=compute_35,code=sm_35 \
                -gencode arch=compute_50,code=sm_50 \
                -gencode arch=compute_50,code=compute_50
BLAS := atlas
ANACONDA_HOME := $(HOME)/anaconda
PYTHON_INCLUDE := $(ANACONDA_HOME)/include \
                $(ANACONDA_HOME)/include/python2.7 \
                $(ANACONDA_HOME)/lib/python2.7/site-packages/numpy/core/include \
PYTHON_LIB := $(ANACONDA_HOME)/lib
INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include
LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib
{% endhighlight %}

Compile :

{% highlight bash %}
#Compile
make all
make test
make runtest

#Launch web server
cd ../digits
./digits-devserver
{% endhighlight %}

and specify `~/caffe` for Caffe path.

#And Theano ?

Have a look if everything is ok with Theano as well :

{% highlight bash %}
sudo apt-get install libopenblas-dev
git clone git://github.com/lisa-lab/DeepLearningTutorials.git
#let's try a logistic regression (http://deeplearning.net/tutorial/logreg.html) on MNIST dataset
python code/logistic_sgd.py
{% endhighlight %}

If GPU is correctly enabled, should be 2 times faster !

**Well done!**
