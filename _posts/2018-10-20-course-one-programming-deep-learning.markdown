---
layout: post
title:  "Course 1: programming deep learning in 5 days only!"
date:   2018-10-20 06:00:00
categories: deep learning
---

You might first check [Course 0: deep learning!](http://christopher5106.github.io/deep/learning/2018/10/20/course-zero-deep-learning.html) if you have not read it.

# Your programming environment

Deep learning requires heavy computations so all deep learning libraries offer the possibility of parallel computing on GPU rather CPU, and distributed computed on multiple GPUs or instances.

While OpenCl (not to confuse with OpenGL or OpenCV) is an open standard for GPU programming, the most used GPU library is CUDA, a private library by NVIDIA, to be used on NVIDIA GPUs only.

CUDNN is a second library coming with CUDA providing you with more optimized operators.

Once installed on your system, these libraries will be called by more high level deep learning frameworks, such as Caffe, Tensorflow, MXNet, CNTK, Torch or Pytorch.

The command `nvidia-smi` enables you to check the status of your GPUs, as with `top` or `ps` commands.

Most recent GPU architectures are Pascal and Volta architectures. The more memory the GPU has, the better. Operations are usually performed with single precision `float16` rather than double precision `float32`, and on new Volta architectures offer Tensor cores specialized with half precision operations.
