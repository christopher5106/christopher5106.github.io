---
layout: post
title:  "Deep learning tutorial on Caffe technology"
date:   2015-09-04 23:00:51
categories: deep learning
---

[Caffe](http://caffe.berkeleyvision.org/) is certainly one of the best frameworks for deep learning, if not the best.

Let's try to put things into order, in order to get a good tutorial :).

###Install

First install Caffe on [Ubuntu]({{ site.url }}/big/data/2015/07/16/deep-learning-install-caffe-cudnn-cuda-for-digits-python-on-ubuntu-14-04.html) or [Mac OS]({{ site.url }}/big/data/2015/07/16/deep-learning-install-caffe-cudnn-cuda-for-digits-python-on-mac-osx.html) with Python layers activated and pycaffe path correctly set `export PYTHONPATH=~/technologies/caffe/python/:$PYTHONPATH`.


###Launch the python shell

In the iPython shell, load the different libraries  :

{% highlight python %}
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
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

###Define a network model

Let's create first a very simple model with a single convolution composed of 3 convolutional neurons, with a kernel of size 5x5 and a stride of 1 :

![simple network]({{ site.url }}/img/simple_network.png)

This net will produce 3 output maps.

The output map for a convolution given receptive field size has a dimension given by the following equation :

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

    net = caffe.Net('examples/net_surgery/conv.prototxt', caffe.TEST)

The names of input layers of the net are given by `print net.inputs`.

The net contains two ordered dictionaries

- `net.blobs` for data  :

    `net.blobs['data']` contains input data, an array  of shape (1, 1, 100, 100)
    `net.blobs['conv']` contains computed data in layer 'conv' (1, 3, 96, 96)

    initialiazed with zeros.

    To print these infos,

        [(k, v.data.shape) for k, v in net.blobs.items()]

- `net.params` a vector of blobs for weight and bias parameters

    `net.params['conv'][0]` contains the weight parameters, an array of shape (3, 1, 5, 5)
    `net.params['conv'][1]` contains the bias parameters, an array of shape (3,)

    initialiazed with 'weight_filler' and 'bias_filler' algorithms.

    To print these infos :

        [(k, v[0].data.shape, v[1].data.shape) for k, v in net.params.items()]


Blobs are a memory abstraction object (depending on the mode), and data is in the field containing the data array :

{% highlight python %}
print net.blobs['conv'].data.shape
{% endhighlight %}

You can draw the network with a simle python command :

    python python/draw_net.py examples/net_surgery/conv.prototxt my_net.png
    open my_net.png

![simple network]({{ site.url }}/img/simple_network.png)

###Compute the network output on an image as input



Let's load a gray image of size 1x360x480 (channel x height x width) :

![Gray cat]({{ site.url }}/img/cat_gray.jpg)

We need reshape the data blob (1, 1, 100, 100) to this new size (1, 1, 360, 480) :

{% highlight python %}
im = np.array(Image.open('examples/images/cat_gray.jpg'))
im_input = im[np.newaxis, np.newaxis, :, :]
net.blobs['data'].reshape(*im_input.shape)
net.blobs['data'].data[...] = im_input
{% endhighlight %}

Let's compute the blobs given this input

{% highlight python %}
net.forward()
{% endhighlight %}

Now `net.blobs['conv']` is filled with data, and the 3 pictures inside each of the 3 neurons (`net.blobs['conv'].data[0,i]`) can be plotted easily.

To save the net parameters `net.params`, just call :

{% highlight python %}
net.save('mymodel.caffemodel')
{% endhighlight %}

###Loading pretrained parameters to classify the image

In the previous net, weight and bias params have been initialiazed randomly.

It's possible to load trained parameters and in this case, the result of the net will produce a classification.

Many models can be downloaded from the community in the [Caffe Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo).

Model informations are written in Github Gist format. The parameters are saved in a `.caffemodel` file specified in the gist. To download the model :

    ./scripts/download_model_from_gist.sh <gist_id>
    ./scripts/download_model_binary.py <dirname>

where <dirname> is the gist directory (by default the gist is saved in the *models* directory).

Let's download the CaffeNet model and the labels corresponding to the classes :

{% highlight bash %}
./scripts/download_model_binary.py models/bvlc_reference_caffenet
./data/ilsvrc12/get_ilsvrc_aux.sh

#have a look at the model
python python/draw_net.py models/bvlc_reference_caffenet/deploy.prototxt caffenet.png
open caffenet.png
{% endhighlight %}

![CaffeNet model]({{ site.url }}/img/caffenet_model.png)

This model has been trained on processed images, so you need to preprocess the image with a preprocessor.

![Cat]({{ site.url }}/img/cat.jpg)

In the python shell :

{% highlight python %}
#load the model
net = caffe.Net('models/bvlc_reference_caffenet/deploy.prototxt',
                'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
                caffe.TEST)

# load input and configure preprocessing
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', np.load('python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))
transformer.set_transpose('data', (2,0,1))
transformer.set_channel_swap('data', (2,1,0))
transformer.set_raw_scale('data', 255.0)

#note we can change the batch size on-the-fly
#since we classify only one image, we change batch size from 10 to 1
net.blobs['data'].reshape(1,3,227,227)

#load the image in the data layer
im = caffe.io.load_image('examples/images/cat.jpg')
net.blobs['data'].data[...] = transformer.preprocess('data', im)

#compute
out = net.forward()

# other possibility : out = net.forward_all(data=np.asarray([transformer.preprocess('data', im)]))

#predicted predicted class
print out['prob'].argmax()

#print predicted labels
labels = np.loadtxt("data/ilsvrc12/synset_words.txt", str, delimiter='\t')
top_k = net.blobs['prob'].data[0].flatten().argsort()[-1:-6:-1]
print labels[top_k]
{% endhighlight %}



###Learn new models: solve the params on training data 

To train a network, you need

- its model definition, as seen previously

- a second protobuf file, the solver file, describing the parameters for the stochastic gradient

Load the solver

    solver = caffe.get_solver('examples/hdf5_classification/nonlinear_solver.prototxt')
    solver = caffe.SGDSolver('models/bvlc_reference_caffenet/solver.prototxt')



###Resources :

[The catalog of available layers](http://caffe.berkeleyvision.org/tutorial/layers.html)

[Create a classification map with net surgery on a trained model](http://localhost:8888/notebooks/examples/net_surgery.ipynb)

![classification map]({{ site.url }}/img/classification_map.png)
