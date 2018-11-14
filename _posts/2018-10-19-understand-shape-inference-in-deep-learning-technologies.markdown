---
layout: post
title:  "Understand shape inference in deep learning technologies"
date:   2018-10-19 00:00:00
categories: deep learning
---

Run the following code in your Python shell with Keras Python installed,

```python
from keras.layers import Input, Embedding
from keras import backend as K
a = Input((12, ))
x = Embedding(1000, 10, trainable=False, input_length=10, name="word_embedding")(a)
print(K.int_shape(x))
print(x.get_shape().as_list())
```

And you'll get suprised. You'll get two different returns:

```
(None, 10, 10)
[None, 12, 10]
```

If you are not sure why, this article is for you!

## Static Shapes

In Tensorflow, the static shape is given by `.get_shape()` method of the Tensor object, which is equivalent to `.shape`.

The static shape is an object of type `tensorflow.python.framework.tensor_shape.TensorShape`.

With `None` instead of an integer, it leaves the possibility for partially defined shapes :

```python
a = Input((None, 10))
print(a.shape)
```

returns `(?, ?, 10)` where unknown dimensions `None` are printed with an question mark **?**. Since we are using the Keras `Input` layer,
the first dimension is systematically `None` for the batch size and cannot be set.

The TensorShape has 2 public attributes, dims and ndims:

```python
print(a.shape.dims) # [Dimension(None), Dimension(None), Dimension(10)]
print(a.shape.ndims) # 3
print(K.ndim(a)) # 3
```

## Dynamic Shapes

Of course, during the run of the graph with input values, all shapes become known. Shape at run time are named *dynamic shapes*.

To access to their values at run time, you can use either the tensorflow operator `tf.shape()` or the Keras wrapper `K.shape()`.

As graph operators, they both return a `tensorflow.python.framework.ops.Tensor` Tensor.

```python
a = Input((None, ))
import numpy as np
f = K.function([a], [K.shape(a)])
print(f( [np.random.random((3,11)) ]))
```

returns `[array([ 3, 11], dtype=int32)]`. 3 is the batch size, and 11 the value for the unknown data dimension at graph definition.

The equivalent of `a.get_shape().as_list()` for static shapes is `tf.unstack(tf.shape(a))`.

The number of dimensions is returned by `tf.rank()` which returns a tensor of rank zero (scalar) and for that reason is very different from `K.ndim` or `ndims` methods with integer return.

## Shape setting for Operators

Most operators have an output shape function that enables to infer the static shape, **without running the graph**, given the shapes of the operators' input tensors.

For example, in Tensorflow you can check how to define the [Shape functions in C++](https://www.tensorflow.org/extend/adding_an_op#shape_functions_in_c) for any operator.

Nevertheless, shape functions do not cover all cases correctly, and in some cases, it is impossible to infer the shape without knowing more of your intent

Let's take an example, where automatic shape inference is not possible.

Let's define a reshaping operation given a reshaping tensor for which the values are not known at graph definition, but only at run time.

Such a reshaping tensor can be for example depending on input tensor dimensions, or any other dynamic shapes:

```python
a = Input((None,))
reshape_tensor = Input((), dtype="int32")
print(a.shape)

# reshape the first input tensor given the second input tensor
x = K.reshape(a, reshape_tensor)
print(x.shape)

# build the model
f = K.function([a, reshape_tensor], [x])

# eval the model on input data
import numpy as np
f([np.random.random((3,10)), [3, 2,5] ])
```

prints

```
(?, ?)
<unknown>
```

The shape of variable `a` is of rank 2, but both dimensions are not known: it is named *partially known shape*. Its value is `TensorShape([None, None])`.

The shape of variable `x` is *unknown*, neither the rank nor the dimensions are known. Its value is `TensorShape(None)`.

That is where, if possible, setting the shape manually can help get a more precise shape than `TensorShape([None, None])` or `TensorShape(None)`.

To set the unknown shape dimensions:

```python
a = Input((None, 10))
a.set_shape((10,10,10))
print(a.shape)
```

Note that shape setting requires to preserve the shape rank and known dimensions. IF I write:

```python
a.set_shape((10,10,10))
```
it leads to a value error

```
ValueError: Shapes (?, ?, 10) and (10, 10, 11) are not compatible
```


Setting the shape enables further operators to compute the shape.

Since Tensorflow does not have a concept of Layers (it is much more based on Operators and Scopes), the `set_shape()` function is the method for shape inference without running the graph.


## Shape setting for Layers

Let's come back to the initial example, where a layer, the `Embedding` layer, is a concept involved in the middle of the Keras graph definition.

The concept of layers gives a struture to the neural networks, enabling to run through the layers later on:

```python
from keras.models import Model
a = Input((11,))
x = Embedding(1000, 10, trainable=False, input_length=10, name="word_embedding")(a)
m = Model(a,x)
for l in m.layers:
    print(l.name)
```

returns the name of each layers: `input_1, word_embedding`.

Still, when a shape cannot be inferred, it is possible to set it also, so that further layers benefit from their output shape information.

Let's see in practice, with a simple custom concatenate Layer.

For the purpose, let me introduce an error in the `compute_output_shape()` function, adding 2 to the last shape dimension, as I did in the `Embedding` layer at the begining of this article, setting `input_length` to 10 instead of 12:


```python
from keras.engine.topology import Layer
class MyConcatenateLayer(Layer):

    def call(self, inputs):
        return K.concatenate(inputs)

    def compute_output_shape(self, input_shape):
        print(input_shape) # [(None, 10), (None, 12)]
        return (None, input_shape[0][1] + input_shape[1][1] + 2)

from keras.layers import Input, Embedding
from keras import backend as K
a = Input((10,))
b = Input((12,))
c = MyConcatenateLayer()([a,b])
print(c.shape) # (?, 22)
print(c._keras_shape) # (?, 24)
```

The code will run without error, as well as the graph evaluation on values for inputs.

As you can see, Keras adds attributes to Tensor such as `_keras_shape`, to be able to retrieve layers information. This can be useful for layer weight saving for example.

The Keras `K.int_shape()` method relies on `_keras_shape` attribute to return the result, leading to propagation of the error.

Since shapes can vary in rank and their values can be `None`, it is difficult, on such a simple concatenation example, to be sure to cover all cases in the shape inference function and this leads to errors.

## A note on CNTK

CNTK distinguishes unknown dimensions into 2 categories:

- the *inferred dimensions* whose value is to be inferred by the system and is printed with **-1** instead of the question mark **?**. For example, in the matrix multiplication *A x B* between tensors A and B, the last dimension of A can be inferred by the system given the first dimension of B. See [here](https://docs.microsoft.com/en-us/cognitive-toolkit/parameters-and-constants#automatic-dimension-inference)

- the *free dimensions* whose value is known only when data is bound to the variable and is printed with **-3**

**Well done!**

Now, you are aware of the Why of the shape setting, its advantages and its risks.
