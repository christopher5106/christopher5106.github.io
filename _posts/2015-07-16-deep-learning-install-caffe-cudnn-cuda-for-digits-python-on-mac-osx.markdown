---
layout: post
title:  "Deep learning with Cuda 7, CuDNN 2 and Caffe for Digits 2 and Python on Mac OS X"
date:   2015-07-16 23:00:51
categories: big data
---

#Install on iMac 27", OS X 10.10.4, NVIDIA GeForce GT 755M 1024 Mo

1\. You need Mac Os Command Line Tools if not already installed :

    xcode-select --install


2\. Install Cuda 7 (pass this step if your Mac GPU is not CUDA capable)

Check your version and the path.

{% highlight bash %}
/usr/local/cuda/bin/nvcc --version
# Cuda compilation tools, release 7.0, V7.0.27
export DYLD_LIBRARY_PATH=/usr/local/cuda/lib:/usr/local/lib:/usr/lib
{% endhighlight %}

You can also install the [latest version of the driver (recommended)](http://www.nvidia.com/object/mac-driver-archive.html) because the driver in Cuda is not the latest one.


3\. Download CuDNN (pass this step if your Mac GPU is not CUDA capable)

    wget https://s3-eu-west-1.amazonaws.com/christopherbourez/public/cudnn-6.5-osx-v2.tgz
    tar xvzf cudnn-6.5-osx-v2.tgz
    rm cudnn-6.5-osx-v2.tgz
    cd cudnn-6.5-osx-v2/
    sudo cp cudnn.h /usr/local/cuda/include/
    sudo cp libcudnn* /usr/local/cuda/lib/


4\. Install the packages

    brew tap homebrew/science
    brew update
    brew install snappy leveldb protobuf gflags glog szip lmdb hdf5 numpy


5\. Install boost 1.57 (Caffe is not compatible with Boost 1.58 as explained [here](http://itinerantbioinformaticist.blogspot.fr/2015/05/caffe-incompatible-with-boost-1580.html)). For that reason change the `/usr/local/Library/Formula/boost.rb` with the contents of [boost.rb 1.57](https://raw.githubusercontent.com/Homebrew/homebrew/6fd6a9b6b2f56139a44dd689d30b7168ac13effb/Library/Formula/boost.rb) and `/usr/local/Library/Formula/boost-python.rb` with the contents of [boost-python.rb 1.57](https://raw.githubusercontent.com/Homebrew/homebrew/3141234b3473717e87f3958d4916fe0ada0baba9/Library/Formula/boost-python.rb).

    brew install boost
    brew install boost-python


6\. Install OpenCV (bundled with numpy and hdf5).

With `brew edit opencv`, change the line

    args << "-DPYTHON_LIBRARY=#{py_lib}/libpython2.7.#{dylib}"

for

    args << "-DPYTHON_LIBRARY=#{py_prefix}/lib/libpython2.7.#{dylib}"

Install:

    brew install opencv


7\. Clone the caffe repository

    git clone https://github.com/BVLC/caffe.git
    cd caffe
    vi Makefile.config

8\. Create the configuration file `Makefile.config`

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
                /usr/lib/python2.7/dist-packages/numpy/core/include
PYTHON_LIB := /usr/lib
PYTHON_INCLUDE += $(dir $(shell python -c 'import numpy.core; print(numpy.core.__file__)'))/include
PYTHON_LIB += $(shell brew --prefix numpy)/lib
INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include
LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib
BUILD_DIR := build
DISTRIBUTE_DIR := distribute
TEST_GPUID := 0
Q ?= @
{% endhighlight %}

If your iMac is not CUDA capable, comment `USE_CUDNN := 1`, `CUDA_DIR := /usr/local/cuda` and `CUDA_ARCH=...` lines and uncomment `CPU_ONLY := 1`

You can verify that Python is at the right place :

{% highlight bash %}
which python
#/usr/local/bin/python
{% endhighlight %}


9\. Build

{% highlight bash %}
make all --jobs=4
make test --jobs=4
make runtest
#python
for req in $(cat python/requirements.txt); do pip install $req; done
make pycaffe
cd ..
{% endhighlight %}

Here is the result of the [runtest run]({{ site.url }}/img/make_runtest_result.txt).

10\. Download DIGITS

{% highlight bash %}
export CUDA_HOME=/usr/local/cuda
git clone https://github.com/NVIDIA/DIGITS.git digits-2.0
cd digits-2.0/digits/
pip install -r requirements.txt
export CAFFE_HOME=../caffe
./digits-devserver
{% endhighlight %}

and choose `../caffe` as Caffe path.

Open Port 5000 on the instance. The server will be running at [http://0.0.0.0:5000/](http://0.0.0.0:5000/)

**Done!**

Troubleshooting :

A few help on common errors :

Undefined symbols for architecture x86_64:   "cv::imread

=> Clean all the opencv before re-installing it

    sudo make uninstall #in the opencv build repo (if you installed from source)
    brew uninstall opencv
    brew uninstall opencv3
    rm /usr/local/lib/libopencv_*
    sudo rm -r /usr/local/share/OpenCV
    sudo rm -r /usr/local/include/opencv2

Be careful to brew messages, all libraries have to be linked.

Undefined symbols for architecture x86_64:
  "google::SetUsageMessage

=> Re-install gflags. `brew install gflags`

Undefined symbols for architecture x86_64:
"leveldb::

=> Re-install leveldb. `brew install leveldb`

Undefined symbols for architecture x86_64: "google::protobuf:

=> Re-install protobuf. `brew install protobuf`

./include/caffe/util/mkl_alternate.hpp:11:10: fatal error: 'cblas.h' file not found
#include <cblas.h>

=> Install Mac OS command line tools `xcode-select --install`.
