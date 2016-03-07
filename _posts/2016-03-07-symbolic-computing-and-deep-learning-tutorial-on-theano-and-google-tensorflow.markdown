---
layout: post
title:  "Symbolic computing and deep learning tutorial with Tensorflow / Theano : 2 for the price of 1"
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

# Symbols in symbolic computing

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

a and b are not classical programming variable, and are named **symbols** or **placeholders** or **tensors**. They are much more like *mathematical variable*. The second advantage of symbolic computing is the **automatic differentiation**, useful to compute gradients. For example let's differentiate the function $$ x \rightarrow x^2 $$ in Theano :

```python
a = th.dscalar()
ga = th.grad(a ** 2,a)
f = theano.function([a], ga)
ga(2)
```

This is great since it does not require to be able to know how to differentiate a function. As you can see, both in Theano and Tensorflow, the ` z = a + b ` is of type Tensor as `a` and `b`.

Tensors are not variables and have no value, but can be evaluated which will launch the operations in a session run as before or directly written :

```python
# in theano
z = a + b
z.eval({a : 10, b : 32})

# in tensorflow
with sess.as_default():
    print z.eval({a: 10, b: 32})

# or z.eval(feed_dict={a: 10, b: 32})
```

Tensors have a type and a shape as in Numpy ([Tensorflow types and shapes](https://www.tensorflow.org/versions/r0.7/resources/dims_types.html) - [Theano types](http://deeplearning.net/software/theano/library/tensor/basic.html)).

# Naming tensors in the graph of operations

Tensorboard, part of Tensorflow, offers us to follow the graph construction state, which is a good idea in parallel of this tutorial. Let's first write the data :

```python
writer = tf.train.SummaryWriter("/tmp/mytutorial_logs", sess.graph_def)
```

and launch Tensorboard :

```
tensorboard --logdir=/tmp/mytutorial_logs
```

Go to [http://localhost:6006/](http://localhost:6006/) under Graph tab to see our first graph :


![]({{ site.url }}/img/tensorflow_tutorial_add.png)


As you can see, our addition operation is present but symbols are not named, it is possible possible to name them in Tensorflow

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
f = theano.function([a, b], a + b, name='addition')
f(10,32)
```

![]({{ site.url }}/img/tensorflow_tutorial_named_add.png)

Tensorflow scopes add a scope prefix to all tensors declared inside the scope, and Tensorboard displays them under a single node that can be expanded in the interface by clicking on it. Scopes can be hierarchical also.

# Share the variables in a first example of network

Let's go a bit further by defining in symbolic computing a first layer of 3 neurons acting on MNIST images (size `28x28=784`, 1 channel) with zero padding, stride 1 and weights initialized with a normal distribution.

![](http://christopher5106.github.io/img/simple_network.png)

As I said about symbolic computation, we use tensors that are abstraction objects of the objets in the memory of the CPU or the GPU, simplifying manipulation, but how can we access their values outside from the result values of a session run ?

For that purpose, Tensorflow created *Variables*, that add an operation to the graph,

```python
x = tf.placeholder(tf.float32, [None, 784])
weights = tf.Variable(tf.random_normal([5, 5, 1, 3], stddev=0.1)))
bias = tf.Variable(tf.constant(0.1, shape=[3]))
z = tf.nn.conv2d(x, weights, strides=[1, 1, 1, 1], padding='SAME') + bias
```

and Theano *shared variables*

```python

```

None
sess.run(tf.initialize_all_variables())


Save the variable


With this simple introduction to symbolic programming, you're now ready to go further and check out [Tensorflow](https://www.tensorflow.org/versions/r0.7/tutorials/index.html) and [Theano](http://deeplearning.net/tutorial/) network examples!

A very nice library that is built on Theano and simplifies the use of Theano is [Lasagne](http://lasagne.readthedocs.org/en/latest/).

**Well done!**
