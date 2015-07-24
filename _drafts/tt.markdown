4\. Download

    brew install libdc1394
    brew install libgphoto2
    brew install ffmpeg
    brew install -v mjpegtools
    brew install gstreamer
    wget https://github.com/Itseez/opencv/archive/3.0.0.zip
    unzip 3.0.0.zip
    rm 3.0.0.zip
    opencv-3.0.0
    mkdir release
    cd release
    [OpenCV 3.0.0](http://opencv.org/downloads.html) and [install](http://docs.opencv.org/doc/tutorials/introduction/linux_install/linux_install.html#linux-installation) with CUDA support following [this very good tutorial](http://www.learnopencv.com/install-opencv-3-on-yosemite-osx-10-10-x/).
    export DYLD_FALLBACK_LIBRARY_PATH=/usr/local/cuda/lib/:$DYLD_FALLBACK_LIBRARY_PATH
    cmake -D WITH_CUDA=ON -D CMAKE_INSTALL_PREFIX=/usr/local -D CMAKE_BUILD_TYPE=RELEASE BUILD_EXAMPLES=ON  -D CMAKE_OSX_ARCHITECTURES=x86_64 ..
    make all
    sudo make install


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




#Install on a AWS g2 instance, with Ubuntu 14.04.

{% highlight bash %}
#Install Cuda
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_7.0-28_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1404_7.0-28_amd64.deb
sudo apt-get update
sudo apt-get -y install cuda
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64
#check everything ok
/usr/local/cuda/bin/nvcc --version
#> Cuda compilation tools, release 7.0, V7.0.27

#Install Cudnn
wget https://s3-eu-west-1.amazonaws.com/christopherbourez/public/cudnn-6.5-linux-x64-v2.tgz
tar xvzf cudnn-6.5-linux-x64-v2.tgz
sudo cp cudnn-6.5-linux-x64-v2/cudnn.h /usr/local/cuda/include/
sudo cp cudnn-6.5-linux-x64-v2/libcudnn* /usr/local/cuda/lib64/

sudo cp cudnn-6.5-linux-x64-v2/cudnn.h /usr/local/cuda-7.0/include/
sudo cp cudnn-6.5-linux-x64-v2/libcudnn* /usr/local/cuda-7.0/lib64/



#Install Git
sudo apt-get -y install git

#Install Caffe
git clone --branch caffe-0.13 https://github.com/NVIDIA/caffe.git
export CAFFE_HOME=~/caffe
cd $CAFFE_HOME


sudo apt-get -y install $APT_FLAGS libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev libgflags-dev libgoogle-glog-dev liblmdb-dev protobuf-compiler libatlas-base-dev
sudo apt-get install -y python-dev python-pip gfortran
sudo apt-get install -y cython python-numpy python-scipy python-skimage python-matplotlib python-h5py python-leveldb python-networkx python-pandas python-dateutil python-protobuf python-gflags python-yaml python-pil
sudo apt-get install -y graphviz
sudo apt-get install -y python-six python-requests python-Flask python-gevent
sudo apt-get -y install --no-install-recommends libboost-all-dev




sudo apt-get -y install python-pip
for req in $(cat python/requirements.txt); do pip install $req; done
sudo pip install boost python-boost

sudo apt-get -y install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev libgflags-dev libgoogle-glog-dev liblmdb-dev protobuf-compiler libatlas-base-dev
sudo apt-get -y install --no-install-recommends libboost-all-dev
sudo apt-get -y install python-dev python-pip python-numpy gfortran
for req in $(cat python/requirements.txt); do sudo pip install $req; done
cp Makefile.config.example Makefile.config
make all --jobs=8
make py
cd ..

#Install Digits
git clone https://github.com/NVIDIA/DIGITS.git digits-2.0
export DIGITS_HOME=~/digits-2.0
cd $DIGITS_HOME
sudo pip install -r requirements.txt
./digits-devserver
{% endhighlight %}

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




#Install Caffe with Ananconda

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
#sudo apt-get remove libopenblas-base
#sudo ldconfig /usr/local/cuda/lib64
#sudo ln /dev/null /dev/raw1394
#conda install protobuf  

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



#And Theano ?

Have a look if everything is ok with Theano as well :

{% highlight bash %}
sudo apt-get install libopenblas-dev
git clone git://github.com/lisa-lab/DeepLearningTutorials.git
#let's try a logistic regression (http://deeplearning.net/tutorial/logreg.html) on MNIST dataset
python code/logistic_sgd.py
{% endhighlight %}

If GPU is correctly enabled, should be 2 times faster !
