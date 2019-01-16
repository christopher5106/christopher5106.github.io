---
layout: post
title:  "Understand batch matrix multiplication"
date:   2018-10-28 05:00:00
categories: deep learning
---

```python
from keras import backend as K
a = K.ones((3,4,5,2))
b = K.ones((2,5,3,7))
c = K.dot(a, b)
print(c.shape)
```

returns

```
ValueError: Dimensions must be equal, but are 2 and 3 for 'MatMul' (op: 'MatMul') with input shapes: [60,2], [3,70].
```

It looks like $$ 60 = 3 \times 4 \times 5 $$ and $$ 70 = 5 \times 3 \times 7 $$.

What's happening ?


### Matrix multiplication when tensors are matrices

The matrix multiplication is performed with `tf.matmul` in Tensorflow or `K.dot` in Keras :

```python
from keras import backend as K
a = K.ones((3,4))
b = K.ones((4,5))
c = K.dot(a, b)
print(c.shape)
```

or

```python
import tensorflow as tf
a = tf.ones((3,4))
b = tf.ones((4,5))
c = tf.matmul(a, b)
print(c.shape)
```

returns a tensor of shape (3,5) in both cases. Simple.


### Keras dot

If I add a dimension:

```python
from keras import backend as K
a = K.ones((2,3,4))
b = K.ones((7,4,5))
c = K.dot(a, b)
print(c.shape)
```

returns a tensor of shape (2, 3, 7, 5).

The matrix multiplication is performed along the 4 values of :

- the last dimension of the first tensor

- the before-last dimension of the second tensor

```python
from keras import backend as K
a = K.ones((1, 2, 3 , 4))
b = K.ones((8, 7, 4, 5))
c = K.dot(a, b)
print(c.shape)
```

returns a tensor of size

- `a.shape` minus last dimension => (1,2,3)

concatenated with

- `b.shape` minus the before last dimension => (8,7,5)

hence : (1, 2, 3, 8, 7, 5)

where each value is given by the formula :

$$ c_{a,b,c,i,j,k} = \sum_r a_{a,b,c,r} b_{i,j, r, k} $$

Not very easy to visualize when ranks of tensors are above 2 :).

**Note that this behavior is specific to Keras dot**. It is a reproduction of Theano behavior.

In particular, it enables to perform a kind of dot product:

```python
from keras import backend as K
a = K.ones((1, 2, 4))
b = K.ones((8, 7, 4, 5))
c = K.dot(a, b)
print(c.shape)
```

returns a tensor of shape (1, 2, 8, 7, 5).


### Batch Matrix Multiplication : tf.matmul or K.batch_dot

There is another operator, `K.batch_dot` that works the same as `tf.matmul`

```python
from keras import backend as K
a = K.ones((9, 8, 7, 4, 2))
b = K.ones((9, 8, 7, 2, 5))
c = K.batch_dot(a, b)
print(c.shape)
```

or

```python
import tensorflow as tf
a = tf.ones((9, 8, 7, 4, 2))
b = tf.ones((9, 8, 7, 2, 5))
c = tf.matmul(a, b)
print(c.shape)
```

returns a tensor of shape (9, 8, 7, 4, 5) in both cases.


$$ c_{a,b,c,i,j} = \sum_r a_{a,b,c,i,r} b_{a,b,c, r, j} $$

So, here the multiplication has been performed considering (9,8,7) as the batch size or equivalent. That could be a position in the image (B,H,W) and for each position we'd like to multiply two matrices.

In CNTK, the same operations will produce an array of dimensions (9, 8, 7, 4, 9, 8, 7, 5) which might not be desired. Here is the trick: in CNTK all operators can be batched, as soon as you declare the first dimension is the batch dimension (dynamic axis) with `C.to_batch()` and batch multiplication could be written this way:

```python
from keras import backend as K
import cntk as C

def cntk_batch_dot(a, b):
    a_shape = K.int_shape(a)
    a = K.reshape(a, [-1] + list(a_shape[-2:]))

    b_shape = K.int_shape(b)
    b = K.reshape(b, [-1] + list(b_shape[-2:]))

    res = C.times(a, b)
    return K.reshape(res, [-1] + list(a_shape[1:4]) + list(b_shape[-1:]))

a = K.ones((9, 8, 7, 4, 2))
b = K.ones((9, 8, 7, 2, 5))
a = C.to_batch(a)
b = C.to_batch(b)
c = cntk_batch_dot(a, b)
print(c.shape)
```

**Well done!**
