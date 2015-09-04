---
layout: post
title:  "Deep learning tutorial on Caffe technology"
date:   2015-09-04 23:00:51
categories: deep learning
---

[Caffe](http://caffe.berkeleyvision.org/) is certainly one of the best frameworks for deep learning, if not the best.

Let's try to put things into order, in order to get a good tutorial :).

####Install

First install Caffe on [Ubuntu]({{ site.url }}/big/data/2015/07/16/deep-learning-install-caffe-cudnn-cuda-for-digits-python-on-ubuntu-14-04.html) or [Mac OS]({{ site.url }}/big/data/2015/07/16/deep-learning-install-caffe-cudnn-cuda-for-digits-python-on-mac-osx.html) with Python layers activated and pycaffe path correctly set `export PYTHONPATH=~/technologies/caffe/python/:$PYTHONPATH`.


####Launch

In the iPython shell, load the different libraries  :

{% highlight python %}
import numpy as np
import matplotlib.pyplot as plt
from PIL  import Image
import caffe
{% endhighlight %}

Set the computation mode CPU

{% highlight python %}
caffe.set_mode_cpu()
{% endhighlight %}

or GPU

{% highlight python %}
caffe.set_device(0)
caffe.set_mode_gpu()
{% endhighlight %}

####Define a model

Let's create first a very simple model with a single convolution composed of 3 convolutional neurons, with a kernel of size 5x5 and a stride of 1 :

![simple network]({{ site.url }}/simple_network.png)


The output map of the convolution given receptive field size has a size following the equation :

    output = (input - kernel_size) / stride + 1


Create a first file `conv.prototxt` describing the neuron network :

      name: "convolution"
      input: "data"
      input_dim: 1
      input_dim: 1
      input_dim: 100
      input_dim: 100
      layer {
        name: "conv"
        type: "Convolution"
        bottom: "data"
        top: "conv"
        convolution_param {
          num_output: 3
          kernel_size: 5
          stride: 1
          weight_filler {
            type: "gaussian"
            std: 0.01
          }
          bias_filler {
            type: "constant"
            value: 0
          }
        }
      }

Load the net

    net = caffe.Net('conv.prototxt', caffe.TEST)

The names of input layers of the net are given by `print net.inputs`.

The net contains two ordered dictionaries

- `net.blobs` for data  :

    `net.blobs['data']`` contains input data, an array  of shape (1, 1, 100, 100)
    `net.blobs['conv']`` contains computed data in layer 'conv' (1, 3, 96, 96)

    initialiazed with zeros.

- `net.params` a vector of blobs for weight and bias parameters

    `net.params['conv'][0]`` contains the weight parameters, an array of shape (3, 1, 5, 5)
    `net.params['conv'][1]`` contains the bias parameters, an array of shape (3,)

    initialiazed with 'weight_filler' and 'bias_filler'.

Blobs are a memory abstraction object (depending on the mode), and data is in the field data

{% highlight python %}
print net.blobs['conv'].data.shape
{% endhighlight %}

You can draw the network with the following python command :

    python python/draw_net.py examples/net_surgery/conv.prototxt my_net.png

####Compute the network on an image

Let's load a gray image (1 channel) of size (height x width) 360x480 and reshape the blob to its new size :

{% highlight python %}
im = np.array(Image.open('examples/images/cat_gray.jpg')) #shape :
im_input = im[np.newaxis, np.newaxis, :, :] #new shape : (1, 1, 360, 480)
net.blobs['data'].reshape(*im_input.shape)
net.blobs['data'].data[...] = im_input
{% endhighlight %}

Let's compute the blobs given this input

    net.forward()

Now `net.blobs['conv']` is not any more a zero and its subpictures can be plotted easily.


####Solve the params

To solve a network, you need a second protobuf file, describing the iterations parameters :



Create the solver file `solv`


Load the

    net = caffe.Net('models/bvlc_reference_caffenet/deploy.prototxt',
                'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
                caffe.TEST)


    solver = caffe.get_solver('examples/hdf5_classification/nonlinear_solver.prototxt')

    and load it in iPython

Save the model :

    net.save('mymodel.caffemodel')


####Resources :

[The catalog of available layers](http://caffe.berkeleyvision.org/tutorial/layers.html)
