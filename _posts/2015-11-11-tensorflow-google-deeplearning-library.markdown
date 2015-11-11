---
layout: post
title:  "Tensorflow : Google Deeplearning Library, let's give it a try"
date:   2015-11-11 23:00:51
categories: deep learning
---

TensorFlow comes as a library such as other deep learning libraries, [Caffe](http://caffe.berkeleyvision.org/), [Theano](http://deeplearning.net/software/theano/), or [Torch](http://torch.ch/). I've already written a lot on Caffe before.

# The main advantage of Tensorflow

Such as Caffe, TensorFlow comes with an [API](http://tensorflow.org/api_docs) in Python and C++.

Python is great for the tasks of experimenting and learning the parameters. These tasks may require tons of data manipulation before you get to the correct prediction model. Remember that 80% of the job of the data scientist is to prepare data, first clean it, then find the correct representation from which a model can learn.

Tensorflow enables GPU execution which is also a very useful tool when it comes to learning, which require usually from a few tens minutes to a few tens hours : learning time will be divided by 5 to 50 on GPU which makes playing with hyperparameters of the model very easy.

Lastly, when the model has been learned, it comes to production, and usually, speed and optimizations become more important and the C++ API is the expected choice in this case.

Tensorflow comes with [Tensorboard](http://tensorflow.org/how_tos/summaries_and_tensorboard/index.md), which looks like the DIGITS interface from NVIDIA :

![Tensorboard](http://api.tensorflow.org/system/image/body/1675/mnist_tensorboard.png)

and graph visualization tool that works like our `python python/draw_net.py` command in Caffe ([Tutorial](http://christopher5106.github.io/deep/learning/2015/09/04/Deep-learning-tutorial-on-Caffe-Technology.html))

![Tensorboard graph](http://api.tensorflow.org/system/image/body/1691/colorby_structure.png)

I would say that Tensorflow brings nothing really new, but **the main advantage is that everything is in one place, and easy to install**, which is very nice. Caffe remains for me the main tool where R&D occurs, but I believe that Tensorflow will become greater and greater in the future. All the work done by Google is very great.


# Let's give a try

Let's install [Tensorflow](http://tensorflow.org/get_started/os_setup.md) on an iMac :

    pip install https://storage.googleapis.com/tensorflow/mac/tensorflow-0.5.0-py2-none-any.whl

You will need protobuf version above 3.0  (otherwise you'll get a `TypeError: __init__() got an unexpected keyword argument 'syntax'`) :

    brew uninstall protobuf
    brew install --devel --build-from-source --with-python -vd protobuf


`--devel` options will enable to install version  'protobuf>=3.0.0a3'.

Let us create a `input_data.py` file](https://tensorflow.googlesource.com/tensorflow/+/master/tensorflow/g3doc/tutorials/mnist/input_data.py).

Let's run the following commands in `ipython` or by creating a file `run.py` that we run with command `python run.py` :

{% highlight python %}
import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
import tensorflow as tf
sess = tf.InteractiveSession()
x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
sess.run(tf.initialize_all_variables())
y = tf.nn.softmax(tf.matmul(x,W) + b)
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
for i in range(1000):
  batch = mnist.train.next_batch(50)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})
{% endhighlight %}

Which gives an accuracy of 0.9092.
