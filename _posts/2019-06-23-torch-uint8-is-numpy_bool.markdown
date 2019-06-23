---
layout: post
title:  "PyTorch Uint8 is Numpy Bool"
date:   2019-06-23 05:00:00
categories: torch
---

A subtlety for array indexing:

In Numpy:

```python
>>> a  = np.arange(10).reshape(2,5)
>>> a
array([[0, 1, 2, 3, 4],
       [5, 6, 7, 8, 9]])

>>> b = a < 8
>>> b
array([[ True,  True,  True,  True,  True],
      [ True,  True,  True, False, False]])

>>> a[b]
array([0, 1, 2, 3, 4, 5, 6, 7])
```

Until now, everything looks fine. And if we use integers, Numpy considers it as indexes:

```python
>>> b=b.astype("uint8")
>>> b
array([[1, 1, 1, 1, 1],
       [1, 1, 1, 0, 0]], dtype=uint8)

>>> b=b.astype(np.uint8)
>>> b
array([[1, 1, 1, 1, 1],
       [1, 1, 1, 0, 0]], dtype=uint8)

>>> a[b]
array([[[5, 6, 7, 8, 9],
       [5, 6, 7, 8, 9],
       [5, 6, 7, 8, 9],
       [5, 6, 7, 8, 9],
       [5, 6, 7, 8, 9]],

      [[5, 6, 7, 8, 9],
       [5, 6, 7, 8, 9],
       [5, 6, 7, 8, 9],
       [0, 1, 2, 3, 4],
       [0, 1, 2, 3, 4]]])

>>> a[b].shape
(2, 5, 5)
```

The integer values 0 and 1 in b array are considered as taking a[0] and a[1] (which are vectors of dimension(5,)). The resulting shape is the shape of b + [5].

In Torch, behavior is very different:

```python
>>> a = torch.arange(10).reshape(2,5)
>>> a
tensor([[0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9]])

>>> b = a < 8
>>> b
tensor([[1, 1, 1, 1, 1],
        [1, 1, 1, 0, 0]], dtype=torch.uint8)
```

The result is in the uint8 type (which replaces boolean in Numpy).

```python
>>> a[b]
tensor([0, 1, 2, 3, 4, 5, 6, 7]
```

Torch requires to use long type for array index:

```python
>>> b = b.to(torch.long)
>>> a[b]
tensor([[[5, 6, 7, 8, 9],
         [5, 6, 7, 8, 9],
         [5, 6, 7, 8, 9],
         [5, 6, 7, 8, 9],
         [5, 6, 7, 8, 9]],

        [[5, 6, 7, 8, 9],
         [5, 6, 7, 8, 9],
         [5, 6, 7, 8, 9],
         [0, 1, 2, 3, 4],
         [0, 1, 2, 3, 4]]])
```

The funny thing in deep learning frameworks is that there are only partly based on Numpy convention... with lot's of subtle differences. One might say we do not speak of arrays, but tensors...

**Sad news**
