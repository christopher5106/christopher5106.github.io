---
layout: post
title:  "Course 1: programming deep learning in 5 days only!"
date:   2018-10-20 06:00:00
categories: deep learning
---

You might first check [Course 0: deep learning!](http://christopher5106.github.io/deep/learning/2018/10/20/course-zero-deep-learning.html) if you have not read it.

In this deep learning course, we'll use Pytorch as deep learning framework, which is the most modern technology in the area... believe me... and we'll explain you later the burden of other technologies.

# Your programming environment

Deep learning demands heavy computations so all deep learning libraries offer the possibility of parallel computing on GPU rather CPU, and distributed computed on multiple GPUs or instances.

The use of specific hardwares such as GPUs requires to install an up-to-date driver in the operating system first.

While OpenCl (not to confuse with OpenGL or OpenCV) is an open standard for GPU programming, the most used GPU library is CUDA, a private library by NVIDIA, to be used on NVIDIA GPUs only.

CUDNN is a second library coming with CUDA providing you with more optimized operators.

Once installed on your system, these libraries will be called by more high level deep learning frameworks, such as Caffe, Tensorflow, MXNet, CNTK, Torch or Pytorch.

The command `nvidia-smi` enables you to check the status of your GPUs, as with `top` or `ps` commands.

Most recent GPU architectures are Pascal and Volta architectures. The more memory the GPU has, the better. Operations are usually performed with single precision `float16` rather than double precision `float32`, and on new Volta architectures offer Tensor cores specialized with half precision operations.

One of the main difficulties come from the fact that different deep learning frameworks are not available and tested on all CUDA versions, CUDNN versions, and even OS. CUDA versions are not available for all driver versions and OS as well. A good solution to adopt is to use Docker files, which limits the choice of the driver version in the operating system: the compliant CUDA and CUDNN versions as well as the deep learning frameworks can be installed inside the Docker container.


# First concept: the batch

When applying the update rule, the best is to compute the gradients on the whole dataset but it is too costly. Usually we use a batch of training examples, it is a trade-off that performs better than a single example and is not too long to compute.

The learning rate needs to be adjusted depending on the batch size. The bigger the batch is, the bigger the learning rate can be.

So, most deep learning programms and frameworks consider the first dimension in your data as the batch size. All other dimensions are the data dimensionality. For an image, it is `BxHxWxC`, written as a shape `(B, H, W, C)`. After a few layers, the shape of the data will change to `(b, h, w, c)` : the batch size remains, the number of channels usually increases $$ c \geq C $$ and the feature map decreases $$ h \leq H, w \leq W$$ for top layers' outputs' shapes.

This format is very common and called *channel last*. Some deep learning frameworks work with *channel first*, such as CNTK, or enables to change the format as in Keras, to `(B, C, W, H)`.


<img src="{{ site.url }}/img/deeplearningcourse/DL14.png">

To distribute the training on multiple GPU or instances, the easiest way is to split along the batch dimension, which we call *data parallellism*, and dispatch the different splits to their respective instance. The parameter update step requires to synchronize more or less the gradient computations. NVIDIA provides fast multi-gpu collectives in its library NCCL, and fast connections between GPUs with NVLINK2.0.


# Second concept: training curves and metrics

As we have seen on [Course 0](http://christopher5106.github.io/deep/learning/2018/10/20/course-zero-deep-learning.html), we use a *cost function* to fit the model to the goal.

So, during training of a model, we usually plot the **training loss**, and if there is no bug, it is not surprising to see it decreasing as the number of training steps or iterations grows.

<img src="{{ site.url }}/img/deeplearningcourse/DL16.png">

Nevertheless, we usually keep 2 to 10 percent of the training set aside from the training process, which we call the **validation dataset** and compute the loss on this set as well. Depending if the model has enough capacity or not, the **validation loss** might increase after a certain step: we call this situation **overfitting**, where the model has too much learned the training dataset, but does not generalize on unseen examples. To avoid this situation to happen, we monitor the validation metrics as well to decide when to stop the training process, after which the model will perform less.

On top of the loss, it is possible to monitor other metrics, such as for example the accuracy. Metrics might not be differentiable, and minimizing the loss might not minimize the metrics. In image classification, a very classical one is the accuracy, that is the ratio of correctly classified examples in the dataset.

We also usually compute the precision/recall curve: precision defines the number of true positive in the examples predicted as positive by the model (true positives + false positives) while the recall is the number of true positives of the total number of positives (true positives + false negatives). While for some applications, such as document retrieval, we prefer to have higher recall, for some other applications, such as automatic document classification, we prefer to have a high precision for automatically classified documents, and leave ambiguities to a human operators. The area under the precision/recall curve (AUC), gives a good estimate of the discrimination quality of our model.

<img src="{{ site.url }}/img/deeplearningcourse/DL11.png">

# Pytorch
