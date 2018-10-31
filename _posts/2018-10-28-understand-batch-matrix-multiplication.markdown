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


### Matrix multiplication

The matrix multiplication is performed with `tf.matmul` in Tensorflow or `K.dot` in Keras :

```python
from keras import backend as K
a = K.ones((3,4))
b = K.ones((4,5))
c = K.dot(a, b)
print(c.shape)
```

returns a tensor of shape (3,5).

If I add a dimension:

```python
from keras import backend as K
a = K.ones((2, 3,4))
b = K.ones((7, 4,5))
c = K.dot(a, b)
print(c.shape)
```

returns a tensor of shape (2, 3, 7, 5).

The matrix multiplication is performed along the 4 values of :

- the last dimension of the first tensor

- the before-last dimension of the second tensor

```python
from keras import backend as K
a = K.ones((1, 2 ,3 , 4))
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

### Batch Matrix Multiplication

There is another operator, `K.batch_dot` that looks

```python
from keras import backend as K
a = K.ones((9, 8, 7, 4, 2))
b = K.ones((9, 8, 7, 2, 5))
c = K.batch_dot(a, b)
print(c.shape)
```

returns a tensor of shape (9, 8, 7, 4, 5).


$$ c_{a,b,c,i,j} = \sum_r a_{a,b,c,i,r} b_{a,b,c, r, j} $$

So, here the multiplication has been performed considering (9,8,7) as the batch size or equivalent. That could be a position in the image (B,H,W) and for each position we'd like to multiply two matrices.

**Well done!**
