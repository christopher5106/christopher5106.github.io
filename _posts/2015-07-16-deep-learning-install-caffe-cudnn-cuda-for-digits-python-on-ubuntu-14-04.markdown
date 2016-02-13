---
layout: post
title:  "Deep learning with Cuda 7, CuDNN 2 and Caffe for Digits 2 and Python on Ubuntu 14.04"
date:   2015-07-16 23:00:51
categories: big data
---
![Classification]({{ site.url }}/img/digits.png)

Install on a AWS g2 instance, with Ubuntu 14.04.

# Install Cuda and Cudnn

{% highlight bash %}
# Install Cuda
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_7.0-28_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1404_7.0-28_amd64.deb
sudo apt-get update
sudo apt-get -y install cuda
rm cuda-repo-ubuntu1404_7.0-28_amd64.deb
#check everything ok
/usr/local/cuda/bin/nvcc --version
#> Cuda compilation tools, release 7.0, V7.0.27

# Install Cudnn
wget https://s3-eu-west-1.amazonaws.com/christopherbourez/public/cudnn-6.5-linux-x64-v2.tgz
tar cudnn-6.5-linux-x64-v2.tgz
rm xvzf cudnn-6.5-linux-x64-v2.tgz
sudo cp cudnn-6.5-linux-x64-v2/cudnn.h /usr/local/cuda-7.5/include/
sudo cp cudnn-6.5-linux-x64-v2/libcudnn* /usr/local/cuda-7.5/lib64/

# Install Git
sudo apt-get -y install git
{% endhighlight %}


# Install Caffe alone

{% highlight bash %}
git clone https://github.com/BVLC/caffe.git
cd caffe
sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler
sudo apt-get install --no-install-recommends libboost-all-dev
sudo apt-get install libgflags-dev libgoogle-glog-dev liblmdb-dev
sudo apt-get install libatlas-base-dev
sudo apt-get install python-dev
sudo apt-get install awscli
vi Makefile.config
{% endhighlight %}

with the following Makefile

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
WITH_PYTHON_LAYER := 1
INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include
LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib
BUILD_DIR := build
DISTRIBUTE_DIR := distribute
TEST_GPUID := 0
Q ?= @
{% endhighlight %}

Compile the code :

{% highlight bash %}
make all -j8
make test
make runtest
sudo ln /dev/null /dev/raw1394
echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH" >> ~/.bashrc
sudo apt-get install linux-image-extra-$(uname -r)
sudo reboot
{% endhighlight %}

# Install Digits with Digits'Caffe...

{% highlight bash %}
wget https://s3-eu-west-1.amazonaws.com/christopherbourez/public/digits-2.0.0-preview.gz
tar xvzf digits-2.0.0-preview.gz
rm xvzf digits-2.0.0-preview.gz
cd digits-2.0/
./install.sh
cd caffe
sudo apt-get -y install --no-install-recommends libboost-all-dev #missing
make all --jobs=8
cd ../digits/
sudo ln /dev/null /dev/raw1394
sudo pip install -r requirements.txt
mkdir /home/ubuntu/data/mnist -p
python tools/download_data/main.py mnist ~/data/mnist
apt-get install linux-image-extra-$(uname -r)
echo '[DIGITS]' >> digits/digits.cfg
echo 'caffe_root = /digits/digits-2.0/caffe' >> digits/digits.cfg
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64 ./digits-server -D
{% endhighlight %}

Open Port 8080 on the instance. The server will be running at [http://0.0.0.0:8080/](http://0.0.0.0:8080/)

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

# ... or simply launch an instance provisionned by Chef

You can launch instantly a g2 instance on AWS with my Chef deployment recipe ['digits-server-simple'](https://github.com/christopher5106/digits-server-simple).

Create a repository containing just one file, named `Berksfile`. In this file,  configure the recipe following the [instructions](https://github.com/christopher5106/digits-server-simple).

Create an AWS Opsworks Stack with your repo and its deploy key. Add to your stack a 'Digits' layer with the recipe, a security group open on port 8080 and an EBS under `/digits` path.

**That's it !** Now, whenever you want DIGITS, simply launch a g2 instance in one click, with the 'Add instance' button and it will configure itself.

# Add a dataset to train on

![Add dataset]({{ site.url }}/img/digits_dataset.png)

and it will create it in the DB

![Classification]({{ site.url }}/img/digits_create_db.png)


# Create the model

![Classification]({{ site.url }}/img/digits_create_model.png)


# The REST API

The Digits classification server offers a great [REST API](https://github.com/NVIDIA/DIGITS/blob/master/docs/API.md) in order to integrate it in a bigger IT process and global system.

![Classification]({{ site.url }}/img/digits_rest.png)

![Classification]({{ site.url }}/img/digits_rest_classification.png)

**Well done!**
