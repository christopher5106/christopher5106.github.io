---
layout: post
title:  "Symbolic computing and deep learning tutorial on Tensorflow and Theano"
date:   2016-03-06 23:00:51
categories: big data
---

I gave a short presentation about [Google Tensorflow](http://christopher5106.github.io/deep/learning/2015/11/11/tensorflow-google-deeplearning-library.html) previously, with install instructions.

Let's put things in order to have a great tutorial with mixed code and explanations and learn **twice faster** with mixed Theano and Tensorflow examples in one tutorial :)

First launch a iPython session and import the two libraries. For Tensorflow, we have to create a session to run the operations in :

```python
import tensorflow as tf
sess = tf.Session()
import theano
import theano.tensor as th
```

Instructions beginning with **tf** will be Tensorflow code, with **th** Theano code.

# Symbolic computation

Symbolic computation is a very different way of programming.

Classical programming defines **variables** that hold values, and operations to modify their values.

In symbolic programming, it's more about building a graph of operations, that will be compiled later for execution. Such an architecture enables the code to be executed indifferently on **CPU** or **GPU** for example.

The second aspect of symbolic computing is that it is much more like *mathematical function*, for example let's define an addition in Tensorflow :

```python
a = tf.placeholder(tf.int8)
b = tf.placeholder(tf.int8)
sess.run(a+b, feed_dict={a: 10, b: 32})
```

and the same in Theano :

```python
a = th.iscalar()
b = th.iscalar()
f = theano.function([a, b], a + b)
f(10,32)
```

a and b are not classical programming variable, and are named **symbols** or **placeholders**. They are much more like *mathematical variable*. The second advantage of symbolic computing is the **automatic differentiation**, useful to compute gradients. For example let's differentiate the function $$ x \rightarrow x^2 $$ in Theano :

```python
a = th.dscalar()
ga = th.grad(a ** 2,a)
f = theano.function([a], ga)
ga(2)
```

This is great since it does not require to be able to know how to differentiate a function.

Let's also follow the graph construction of the tutorial in Tensorboard :

```python
writer = tf.train.SummaryWriter("/tmp/mytutorial_logs", sess.graph_def)
```

and launch Tensorboard :

```
tensorboard --logdir=/tmp/mytutorial_logs
```

and go to [http://localhost:6006/](http://localhost:6006/) under Graph tab to see our first graph :


![]({{ site.url }}/img/tensorflow_tutorial_add.png)


As you can see, our symbols are not named, it is possible possible to name them in Tensorflow

```python
a = tf.placeholder(tf.int8, name="a")
b = tf.placeholder(tf.int8, name="b")

with tf.name_scope("addition") as scope:
  my_addition = a+b

sess.run(my_addition, feed_dict={a: 10, b: 32})
writer = tf.train.SummaryWriter("/tmp/mytutorial_logs", sess.graph_def)
```

or in Theano for debugging :

```python
a = th.iscalar('a')
b = th.iscalar('b')
f = theano.function([a, b], a + b)
f(10,32)
```

![]({{ site.url }}/img/tensorflow_tutorial_named_add.png)

# Define a first network

![](http://christopher5106.github.io/img/simple_network.png)
