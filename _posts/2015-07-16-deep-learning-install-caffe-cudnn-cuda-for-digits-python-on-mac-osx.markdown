---
layout: post
title:  "Deep learning with Cuda 7, CuDNN 2 and Caffe for Digits 2 and Python on iMac with NVIDIA GeForce 755M GPU (Mac OS X)"
date:   2015-07-16 23:00:51
categories: big data
---

#Install on iMac 27", OS X 10.10.4, NVIDIA GeForce GT 755M 1024 Mo

[![Classification]({{ site.url }}/img/mac_digits3.png)]({{ site.url }}/img/mac_digits.png)

1\. Install Mac Os Command Line Tools if not already installed :

    xcode-select --install


2\. Install Cuda 7 (pass this step if your Mac GPU is not CUDA capable)

Check your version and the path.

{% highlight bash %}
/usr/local/cuda/bin/nvcc --version
# Cuda compilation tools, release 7.0, V7.0.27
export DYLD_LIBRARY_PATH=/usr/local/cuda/lib
{% endhighlight %}

You can also install the [latest version of the driver (recommended)](http://www.nvidia.com/object/mac-driver-archive.html) because the driver in Cuda is not the latest one.


3\. Download CuDNN (pass this step if your Mac GPU is not CUDA capable)

    wget https://s3-eu-west-1.amazonaws.com/christopherbourez/public/cudnn-6.5-osx-v2.tgz
    tar xvzf cudnn-6.5-osx-v2.tgz
    rm cudnn-6.5-osx-v2.tgz
    cd cudnn-6.5-osx-v2/
    sudo cp cudnn.h /Developer/NVIDIA/CUDA-7.0/include/
    sudo cp libcudnn* /usr/local/cuda/lib/


4\. Install Python packages

    brew install python #if not already installed
    pip install --upgrade pip setuptools #or update

You can verify that Python is at the right place, installed via Homebrew :

{% highlight bash %}
which python
#> /usr/local/bin/python
{% endhighlight %}


5\. Install other packages

    brew tap homebrew/science
    brew update
    brew install snappy leveldb gflags glog szip lmdb hdf5 numpy opencv graphviz
    brew install --build-from-source --with-python -vd protobuf


6\. Install boost 1.57 (Caffe is not compatible with Boost 1.58 as explained [here](http://itinerantbioinformaticist.blogspot.fr/2015/05/caffe-incompatible-with-boost-1580.html)). For that reason change the `/usr/local/Library/Formula/boost.rb` with the contents of [boost.rb 1.57](https://raw.githubusercontent.com/Homebrew/homebrew/6fd6a9b6b2f56139a44dd689d30b7168ac13effb/Library/Formula/boost.rb) and `/usr/local/Library/Formula/boost-python.rb` with the contents of [boost-python.rb 1.57](https://raw.githubusercontent.com/Homebrew/homebrew/3141234b3473717e87f3958d4916fe0ada0baba9/Library/Formula/boost-python.rb).

    brew install --build-from-source -vd boost boost-python


7\. Clone the caffe repository

    mkdir technologies
    cd technologies
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
PYTHON_LIB := /usr/local/Cellar/python/2.7.10_2/Frameworks/Python.framework/Versions/2.7/lib /usr/lib
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

Here is the full copy of the [Makefile.config]({{ site.url }}/img/Makefile.config).

9\. Build

{% highlight bash %}
export DYLD_LIBRARY_PATH=/usr/local/cuda/lib
make all --jobs=4
make test --jobs=4
make runtest
#python
for req in $(cat python/requirements.txt); do pip install $req; done
make pycaffe
export PYTHONPATH=~/technologies/caffe/python/:$PYTHONPATH
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

The server will be running at [http://0.0.0.0:5000/](http://0.0.0.0:5000/)

You can then [have fun with DIGITS as we did on Ubuntu]({{ site.url }}/big/data/2015/07/16/deep-learning-install-caffe-cudnn-cuda-for-digits-python-on-ubuntu-14-04.html) : download the MNIST dataset and train a first classifier on your GPU.

[![Classification]({{ site.url }}/img/mac_digits_create_dataset3.png)]({{ site.url }}/img/mac_digits_create_dataset.png)

[![Classification]({{ site.url }}/img/mac_digits_learn3.png)]({{ site.url }}/img/mac_digits_learn.png)


**Well done!**

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

=> Re-install protobuf. `brew install --build-from-source --with-python -vd protobuf`

./include/caffe/util/mkl_alternate.hpp:11:10: fatal error: 'cblas.h' file not found
#include <cblas.h>

=> Install Mac OS command line tools `xcode-select --install`.


TEXT                 0000000109e08000-0000000109e0a000 [    8K] r-x/rwx SM=COW  /usr/local/Cellar/python/2.7.10_2/Frameworks/Python.framework/Versions/2.7/Resources/Python.app/Contents/MacOS/Python

Thread 0 Crashed:: Dispatch queue: com.apple.main-thread
0   ???                           	000000000000000000 0 + 0
1   org.python.python             	0x0000000112c7f0dd PyEval_GetGlobals + 23
2   org.python.python             	0x0000000112c8e62b PyImport_Import + 137
3   org.python.python             	0x0000000112c8cd27 PyImport_ImportModule + 31
4   caffe.so                     	0x000000010c92bcf8 caffe::init_module__caffe() + 4328
5   libboost_python.dylib         	0x0000000112ba0391 boost::python::handle_exception_impl(boost::function0<void>) + 81
6   libboost_python.dylib         	0x0000000112ba13b9 boost::python::detail::init_module(char const*, void ()()) + 121

=> Add `/usr/local/Cellar/python/2.7.10_2/Frameworks/Python.framework/Versions/2.7/lib` to the PYTHON lib in the Makefile.


nvcc fatal : The version ('60100') of the host compiler ('Apple clang') is not supported

=> Download XCode 6.0 to replace 6.3
Have a look at [compatibilities](http://docs.nvidia.com/cuda/cuda-getting-started-guide-for-mac-os-x/#axzz3hb15JIpL) and [Clang versions](https://gist.github.com/yamaya/2924292).
