---
layout: post
title:  "Course 2: build deep learning neural networks in 5 days only!"
date:   2018-10-20 08:00:00
categories: deep learning
---

Here is my course of deep learning in 5 days only!

You might first check [Course 0: deep learning!](http://christopher5106.github.io/deep/learning/2018/10/20/course-zero-deep-learning.html) and [Course 1: program deep learning!](http://christopher5106.github.io/deep/learning/2018/10/20/course-one-programming-deep-learning.html) if you have not read them.


# Image classification

I'll present some architectures of neural networks for computer vision.

#### AlexNet (2012)

The network was made up of 5 convolution layers, max-pooling layers, dropout layers, and 3 fully connected layers (60 million parameters and 500,000 neurons).

It was the first neural network to outperform the state of the art image classification of that time and it won the 2012 ILSVRC (ImageNet Large-Scale Visual Recognition Challenge).

<img src="{{ site.url }}/img/deeplearningcourse/DL44.png">


Main Points:

- Used ReLU (Rectified Linear Unit) for the nonlinearity functions (Found to decrease training time as ReLUs are several times faster than the conventional tanh function).

- Used data augmentation techniques that consisted of image translations, horizontal reflections, and patch extractions.

- Implemented dropout layers in order to combat the problem of overfitting to the training data.

- Trained the model using batch stochastic gradient descent, with specific values for momentum and weight decay.

####	VGGNet (2014)

VGG neural architecture reduced the size of each layer but increased the overall depth of the network (up to 16 - 19 layers) and reinforced the idea that convolutional neural networks have to be deep in order to work well on visual data.

It finished at the first and the second places in the localisation and classification tasks respectively at 2014 ILSVRC.

<img src="{{ site.url }}/img/deeplearningcourse/DL48.png">


Main Points

- The use of only 3x3 sized filters is quite different from AlexNet’s 11x11 filters in the first layer: the combination of two 3x3 conv layers has an effective receptive field of 5x5.

- It simulates a larger filter while decreasing in the number of parameters.

- As the spatial size of the input volumes at each layer decrease (result of the conv and pool layers), the depth of the volumes increase due to the increased number of filters as you go down the network.

- Interesting to notice that the number of filters doubles after each maxpool layer. This reinforces the idea of shrinking spatial dimensions, but growing depth.

####	GoogLeNet and the Inception (2015)

GoogLeNet was one of the first models that introduced the idea that CNN layers didn’t always have to be stacked up sequentially and creating smaller networks or "modules" :

<img src="{{ site.url }}/img/deeplearningcourse/DL45.png">

that could be repeated inside the network:

<img src="{{ site.url }}/img/deeplearningcourse/DL45.jpg">

GoogLeNet is a 22 layer CNN and was the winner of ILSVRC 2014

Main Points

- Used 9 Inception modules in the whole architecture, with over 100 layers in total!

- No use of fully connected layers : it use an average pool instead, to go from a 7x7x1024 volume to a 1x1x1024 volume. This saves a huge number of parameters.

- Uses 12x fewer parameters than AlexNet.

- During testing, multiple crops of the same image are fed into the network, and the softmax probabilities are averaged to give the final prediction.

- There are updated versions of the Inception module.

####	ResNet (2015)

The central idea of Residual Nets is to reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions :

$$ x \rightarrow x + R(x) $$

where R is a residual to learn.

Residuals are stacks of convolution.

<img src="{{ site.url }}/img/deeplearningcourse/DL46.png">

That structure is then sequentially stacked several tenth of time (or more).

<img src="{{ site.url }}/img/deeplearningcourse/DL47.png">

The winner of the classification task of ILSVRC 2015 is a ResNet network.

Main Points

- On the ImageNet dataset residual nets of a depth of up to 152 layers were used being 8x deeper than VGG nets while still having lower complexity.

- Residual networks are easier to optimize than traditional networks and can gain accuracy from considerably increased depth. In fact, they do not suffer from the vanishing gradients when the depth is too important.

- Residual networks have a natural limit : a 1202-layer network was trained but got a lower test accuracy, presumably due to overfitting.


####	SqueezeNet (2016)

Up to this point, research focused on improving the accuracy of neural networks, the SqueezeNet team took the path of designing smaller models while maintaining the accuracy unchanged.

It resulted into SqueezeNet, which is a convolutional neural network architecture that has 50 times fewer parameters than AlexNet while maintaining its accuracy on ImageNet.

3 main strategies are used while designing that new architecture:

- Replace the majority of 3x3 filters with 1x1 filters (to fit within a budget of a certain number of convolution filters).

- Decrease the number of input channels to 3x3 filters, using dedicated filters names squeeze layers.

- Downsample late in the network so that convolution layers have large activation maps.

The first 2 strategies are about judiciously decreasing the quantity of parameters in a CNN while attempting to preserve accuracy.

The last one is about maximizing accuracy on a limited budget of parameters.

<img src="{{ site.url }}/img/deeplearningcourse/DL46.jpg">


####	PreActResNet (2016)

PreActResNet stands for Pre-Activation Residuel Net and is an evolution of ResNet described above.

The residual unit structure is changed from (a) to (b) (BN: Batch Normalization) :

<img src="{{ site.url }}/img/deeplearningcourse/DL49.png">

The activation functions ReLU and BN are now seen as “pre-activation” of the weight layers, in contrast to conventional view of “post-activation” of the weighted output.

This changes allowed to increase further the depth of the network while improving its accuracy on ImageNet for instance.


####	DenseNet (2016)

DenseNet is a network architecture where each layer is directly connected to every other layer in a feed-forward fashion (within each dense block). For each layer:

- the feature maps of all preceding layers are treated as separate inputs,

- its own feature maps are passed on as inputs to all subsequent layers.

<img src="{{ site.url }}/img/deeplearningcourse/DL49.jpeg">

This connectivity pattern yields on CIFAR10/100 accuracies as good as its predecessors (with or without data augmentation) and SVHN. On the large scale ILSVRC 2012 (ImageNet) dataset, DenseNet achieves a similar accuracy as ResNet, but using less than half the amount of parameters and roughly half the number of FLOPs.


####	ResNeXt (2016)

The neural network architecture ResNeXt is based upon 2 strategies :

- stacking building blocks of the same shape (strategy inherited from VGG and ResNet)

- the “split-transform-merge” strategy that is derived from the Inception models and all its variations (split the input, transform it, merge the transformed signal with the original input).

That design exposes a new dimension, called "cardinality" (the size of the set of parallel transformations), as an essential factor in addition to the dimensions of depth and width.

<img src="{{ site.url }}/img/deeplearningcourse/DL51.png">

It is empirically observed that :

- when keeping the complexity constant, increasing cardinality improves classification accuracy.

- increasing cardinality is more effective than going deeper or wider when we increase the capacity.

ResNeXt finished at the second place in the classification task at 2016 ILSVRC.

<img src="{{ site.url }}/img/deeplearningcourse/DL50.png">



####	DPN (2017)

DPN is a family of convolutional neural networks that intends to efficiently merge residual networks and densely connected networks to get the benefits of both architecture:

- residual networks implicitly reuse features, but it is not good at exploring new ones,

- densely connected networks keep exploring new features but suffers from higher redundancy.

<img src="{{ site.url }}/img/deeplearningcourse/DL52.png">

Consequently, DPN (Dual Path Network) contains 2 path :

- a residual alike path (green path, similar to the identity function),

- a densely connected alike path (blue path, similar to a dense connection within each dense block).



#### NASNet (2017)

NASNet learns to architect a the neural network itself, by the Neural Architecture Search (NAS) framework, which uses a reinforcement learning search method to optimize architecture configuration.

<img src="{{ site.url }}/img/deeplearningcourse/DL53.png">

Each network proposed by the RI network is further trained and its accuracy is our reward. To keep the computational effort affordable, the training is performed on a subset of the complete dataset.

The key principle of this model is the design of a new search space (named the "NASNet search space") which enables transferability from the smallest dataset to the complete one.

Two new modules have been found to achieve state-of-the-art accuracy.

<img src="{{ site.url }}/img/deeplearningcourse/DL54.png">


#### SENet (2017)

Typical convolutional neural networks builds a description of its input image by progressively capturing patterns in each of its layers. For each of them, a set of filters are learnt to express local spatial connectivity patterns from its input channels. Convolutional filters captures informative combinations by fusing spatial and channel-wise information together within local receptive fields.

SENet (Squeeze-and-Excitation Networks) focuses on the relation between channels and recalibrates at transformation step its features so that informative features are emphazised and less useful ones suppressed (independently of their spatial location).

To do so, SENet uses a new architectural unit that consists of 2 steps :

First, squeeze the block input (typically the output of any other convolutional layer) to create a global information using global average pooling,

Then, “excite” the most informative features using adaptive recalibration.

The adaptative recalibration is done as follows :

reduce the dimension of its input using a fully connected layer (noted FC below),

go through a non-linearity function (ReLU function),

restore the dimension of its data using another fully connected layer,

use sigmoid function to transform each output into a scale parameter between 0 and 1,

linearly rescale each original input of the SE unit according to the scale parameters.


X : input of the block that will be enhanced by the squeeze and excitation method Ftr : original convolutional operator to be enhanced

U  : output of Ftr

Fsq   : squeeze function

Fex   : excitation function (creates the scaling parameters)

Fscale : scaling function (scale the original output of Ftr according to the SENet calibration ouput)

X̃  : recalibrated Ftr output

Below are 2 examples of existing blocks enhanced with an SE unit:


The winner of the classification task of ILSVRC 2017 is a modified ResNeXt integrating SE blocks.

#### MobileNet v1/v2

There has been also recently some effort to adapt neural networks to less powerfull architecture such as mobile devices, leading to the creation of a class of networks names MobileNet.

The diagram below illustrates that accepting a (slightly) lower accuracy than the state of the art, it is possible to create networks much less demanding in terms of resources (note that the multiply/add axis is on a logarithmic scale).

However since this study focuses on state of the art performances, those networks are not studied further.




# Under construction

a recurrent network is a feedforward network with two inputs
<img src="{{ site.url }}/img/deeplearningcourse/DL20.png">
hidden information


<img src="{{ site.url }}/img/deeplearningcourse/DL21.png">


convolutions
<img src="{{ site.url }}/img/deeplearningcourse/DL24.png">

<img src="{{ site.url }}/img/deeplearningcourse/DL25.png">

<img src="{{ site.url }}/img/deeplearningcourse/DL26.png">

dilated conv
<img src="{{ site.url }}/img/deeplearningcourse/DL29.png">

Global averaging

<img src="{{ site.url }}/img/deeplearningcourse/DL27.png">


Max pooling
<img src="{{ site.url }}/img/deeplearningcourse/DL28.png">


Batch normalization
statistics at the output are
variance 1, mean 0
training layers on statistics not changning
learning a scale and a bias after normalization
