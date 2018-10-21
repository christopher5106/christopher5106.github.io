---
layout: post
title:  "Course 1: programming deep learning in 5 days only!"
date:   2018-10-20 06:00:00
categories: deep learning
---

You might first check [Course 0: deep learning!](http://christopher5106.github.io/deep/learning/2018/10/20/course-zero-deep-learning.html) if you have not read it.

In this deep learning course, we'll use Pytorch as deep learning framework, the most modern technology in the area.

# Your programming environment

Deep learning demands heavy computations so all deep learning libraries offer the possibility of parallel computing on GPU rather CPU, and distributed computed on multiple GPUs or instances.

The use of specific hardwares such as GPUs requires to install an up-to-date driver in the operating system first.

While OpenCl (not to confuse with OpenGL or OpenCV) is an open standard for GPU programming, the most used GPU library is CUDA, a private library by NVIDIA, to be used on NVIDIA GPUs only.

CUDNN is a second library coming with CUDA providing you with more optimized operators.

Once installed on your system, these libraries will be called by more high level deep learning frameworks, such as Caffe, Tensorflow, MXNet, CNTK, Torch or Pytorch.

The command `nvidia-smi` enables you to check the status of your GPUs, as with `top` or `ps` commands.

Most recent GPU architectures are Pascal and Volta architectures. The more memory the GPU has, the better. Operations are usually performed with single precision `float16` rather than double precision `float32`, and on new Volta architectures offer Tensor cores specialized with half precision operations.

One of the main difficulties come from the fact that different deep learning frameworks are not available and tested on all CUDA versions, CUDNN versions, and even OS. CUDA versions are not available for all driver versions and OS as well. A good solution to adopt is to use Docker files, which limits the choice of the driver version in the operating system: the compliant CUDA and CUDNN versions as well as the deep learning frameworks can be installed inside the Docker container.


# The batch

When applying the update rule, the best is to compute the gradients on the whole dataset but it is too costly. Usually we use a batch of training examples, it is a trade-off that performs better than a single example and is not too long to compute.

The learning rate needs to be adjusted depending on the batch size. The bigger the batch is, the bigger the learning rate can be.

So, most deep learning programms and frameworks consider the first dimension in your data as the batch size. All other dimensions are the data dimensionality. For an image, it is `BxHxWxC`, written as a shape `(B, H, W, C)`. After a few layers, the shape of the data will change to `(b, h, w, c)` : the batch size remains, the number of channels usually increases $$ c \geq C $$ and the feature map decreases $$ h \leq H, w \leq W$$ for top layers' outputs' shapes.

This format is very common and called *channel last*. Some deep learning frameworks work with *channel first*, such as CNTK, or enables to change the format as in Keras, to `(B, C, W, H)`.


<img src="{{ site.url }}/img/deeplearningcourse/DL14.png">

To distribute the training on multiple GPU or instances, the easiest way is to split along the batch dimension, which we call *data parallellism*, and dispatch the different splits to their respective instance. The parameter update step requires to synchronize more or less the gradient computations. NVIDIA provides fast multi-gpu collectives in its library NCCL, and fast connections between GPUs with NVLINK2.0.


# Training curves and metrics

As we have seen on [Course 0](http://christopher5106.github.io/deep/learning/2018/10/20/course-zero-deep-learning.html), we use a *cost function* to fit the model to the goal.

So, during training of a model, we usually plot the **training loss**, and if there is no bug, it is not surprising to see it decreasing as the number of training steps or iterations grows.

<img src="{{ site.url }}/img/deeplearningcourse/DL16.png">

Nevertheless, we usually keep 2 to 10 percent of the training set aside from the training process, which we call the **validation dataset** and compute the loss on this set as well. Depending if the model has enough capacity or not, the **validation loss** might increase after a certain step: we call this situation **overfitting**, where the model has too much learned the training dataset, but does not generalize on unseen examples. To avoid this situation to happen, we monitor the validation metrics as well to decide when to stop the training process, after which the model will perform less.

On top of the loss, it is possible to monitor other metrics, such as for example the accuracy. Metrics might not be differentiable, and minimizing the loss might not minimize the metrics. In image classification, a very classical one is the accuracy, that is the ratio of correctly classified examples in the dataset.

We also usually compute the precision/recall curve: precision defines the number of true positive in the examples predicted as positive by the model (true positives + false positives) while the recall is the number of true positives of the total number of positives (true positives + false negatives). While for some applications, such as document retrieval, we prefer to have higher recall, for some other applications, such as automatic document classification, we prefer to have a high precision for automatically classified documents, and leave ambiguities to a human operators. The area under the precision/recall curve (AUC), gives a good estimate of the discrimination quality of our model.

<img src="{{ site.url }}/img/deeplearningcourse/DL11.png">


# Programming deep learning

A deep learning library offers the following characteristics :

1. It works very well with Numpy arrays. Arrays are called **Tensors**. Moreover, operations on Tensors follow lot's of Numpy conventions.

2. It provides abstract classes, such as Tensors, for parallel computing on GPU rather CPU, and distributed computing on multiple GPUs or instances, since Deep learning demands huge computations.

3. Operators have a 'backward' implementation, computing the gradients for you, with respect to the inputs or parameters.

Let's load Pytorch Python module into a Python shell, as well as Numpy library, check the Pytorch version is correct and the Cuda library is correctly installed:

```python
import torch
import numpy as np
print(torch.__version__) # 0.4.1
print(torch.cuda.is_available()) # True
```

#### 1. Numpy compatibility

You can easily check the following commands in Pytorch and Numpy:

| ####### Command ####### | Numpy  | Pytorch  |
|---|---|---|
| 5x3 matrix, uninitialized  |  x = np.empty((5,3)) | x = torch.Tensor(5,3)  |
| initialized with ones | x = np.ones((5,3)) | x = torch.ones(5,3) |
| initialized with zeros | x = np.zeros((5,3)) | x = torch.zeros(5,3) |
| randomly initialized matrix  |  x = np.random.random((5,3)) | x = torch.rand(5, 3)  |
| Shape/size  |  x.shape | x.size  |
| Addition | +/np.add | +/torch.add |
| In-place addition | x+= | x.add_() |
| First column | x[:, 1] | x[:, 1] |

You can link Numpy array and Torch Tensor, either with

```python
numpy_array = np.ones(5)
torch_array = torch.from_numpy(numpy_array)
```

or

```python
torch_array = torch.ones(5)
numpy_array = torch_array.numpy()
```

which will keep the pointers to the same values:

```python
numpy_array+= 1
print(torch_array) # tensor([2., 2., 2., 2., 2.])
torch_array.add_(1)
print(numpy_array) # [3. 3. 3. 3. 3.]
```

#### 2. GPU computing

It is possible to transfer tensors values between RAM and each GPUs:

```python
a = torch.ones(5,) # tensor([1., 1., 1., 1., 1.])
b = torch.ones(5,)
a_ = a.cuda() # tensor([1., 1., 1., 1., 1.], device='cuda:0')
b_ = b.cuda()
x = a_ + b_ # tensor([2., 2., 2., 2., 2.], device='cuda:0')
y = x + 1 # tensor([3., 3., 3., 3., 3.], device='cuda:0')
z = y.cuda(1) # tensor([3., 3., 3., 3., 3.], device='cuda:1')
```

but keep in mind that synchronization is lost (contrary to Numpy Arrays and Torch Tensors):

```python
a.add_(1)
print(a_) # tensor([1., 1., 1., 1., 1.], device='cuda:0')
```

Tensors behave as classical programming non-reference variables and their content is copied.


#### 3. Automatic differentiation




# Perform a gradient descent on a classification problem



We do it by ourself without packages.



use of cross entropy

Toy dataset of values between 0 and 1.

<img src="{{ site.url }}/img/deeplearningcourse/DL42.png">

Take an ensemble

<img src="{{ site.url }}/img/deeplearningcourse/DL18.png">


<img src="{{ site.url }}/img/deeplearningcourse/DL15.png">


The art of choosing the learning rate
<img src="{{ site.url }}/img/deeplearningcourse/DL20.png">


The art of choosing initialization
small variance, positive and negative values

<img src="{{ site.url }}/img/deeplearningcourse/DL30.png">

# Program with modules

A module is an object to learn specifically designed for deep learning neural networks.

A layer is a module:
<img src="{{ site.url }}/img/deeplearningcourse/DL31.png">
because it has weights and a forward function.

The composition of modules makes a module:

<img src="{{ site.url }}/img/deeplearningcourse/DL30.png">

The modules help organize layers and reuse their definitions.



1- rewrite the model as module using nn modules
2- rewrite training loop using the optim package (zeroing gradients + applying the gradients with an update rule)
3- look at different update rules
4- plot the training curves (loss,...)
5- gpu


view reshape


**Well done!**

Now let's go to next course: [Course 2: building deep learning networks!](http://christopher5106.github.io/deep/learning/2018/10/20/course-two-build-deep-learning-networks.html)
