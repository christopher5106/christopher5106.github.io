---
layout: post
title:  "Tensorflow : Google Deeplearning Library, let's give a try"
date:   2015-11-11 23:00:51
categories: deep learning
---


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
