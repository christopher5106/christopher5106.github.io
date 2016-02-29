---
layout: post
title:  "Compile Caffe on Windows with Cygwin"
date:   2016-02-29 23:00:51
categories: big data
---


First install [Cygwin](https://cygwin.com/install.html) with following packages :

- devel > gcc-g++ 5.3.0-3

- devel > make 4.1_1

- devel > cmake 3.3.2-1

- libs > libprotobuf-devel 2.5.0-1

- libboost-devel 1.58.0-1

- libboost_python 1.58.0-1

- libopencv-devel


Download the files :




In Cygwin terminal :


```
git clone https://github.com/BVLC/caffe.git
cd caffe
cp Makefile.config.example Makefile.config
```
Edit the file so :



Compile gflags :

```
mkdir 3rdparty/include/gflags
git clone https://github.com/gflags/gflags.git
cd gflags
mkdir build && cd build
ccmake ..
make
cp include/* ../../3rdparty/include/gflags/
cp lib/* ../../3rparty/lib/
cd ../..
git clone https://github.com/google/glog.git
cd glog


```


Compile Caffe :

    make
