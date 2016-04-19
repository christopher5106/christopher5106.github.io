---
layout: post
title:  "Supervised learning, unsupervised learning with Spatial Transformer Networks tutorial in Caffe and Tensorflow : improve document classifier and character reading"
date:   2016-04-18 23:00:51
categories: big data
---


# Spatial Transformer Networks

Spatial Transformer Networks (SPN) is a network invented by [Max Jaderberg, Karen Simonyan, Andrew Zisserman, Koray Kavukcuoglu](http://arxiv.org/pdf/1506.02025v3.pdf) at Google DeepMind artifical intelligence lab in London.

The use of a SPN is

- to improve classification

- to subsample an input

- to learn subparts of objects

- to locate objects in an image without supervision


# The maths

**SPN predicts the coefficients of an affine transformation** :

![]({{ site.url }}/img/spatial_transformer_networks.png)

If $$ (x,y) $$ are normalized coordinates, $$ (x,y) \in [-1,1] \times [-1,1]  $$, an affine transformation is given by a matrix multiplication :

$$

\left( \begin{array}{c} x_{in} \\ y_{in} \end{array} \right) =

\left[ \begin{array}{cc} \theta_{11} & \theta_{12} & \theta_{13} \\ \theta_{21} & \theta_{22} & \theta_{23} \end{array} \right]

\left( \begin{array}{c} x_{out} \\ y_{out} \\ 1 \end{array} \right)

$$

A simple translation by $$ (t_x,t_y) $$ would be

$$

\left( \begin{array}{c} x_{in} \\ y_{in} \end{array} \right) =

\left[ \begin{array}{cc} 0 & 0 & t_{x} \\ 0 & 0 & t_{y} \end{array} \right]

\left( \begin{array}{c} x_{out} \\ y_{out} \\ 1 \end{array} \right)

$$

An isotropic scaling by a factor s would be

$$

\left( \begin{array}{c} x_{in} \\ y_{in} \end{array} \right) =

\left[ \begin{array}{cc} s & 0 & 0 \\ 0 & s & 0  \end{array} \right]

\left( \begin{array}{c} x_{out} \\ y_{out} \\ 1 \end{array} \right)

$$

For a clockwise rotation of angle $$ \alpha $$

$$

\left( \begin{array}{c} x_{in} \\ y_{in} \end{array} \right) =

\left[ \begin{array}{cc} cos \ \alpha & - sin \ \alpha & 0 \\ sin \ \alpha & cos \ \alpha & 0  \end{array} \right]

\left( \begin{array}{c} x_{out} \\ y_{out} \\ 1 \end{array} \right)

$$

The global case for a clockwise rotation of angle $$ \alpha $$, a scaling by a factor s, and a translation of the center of $$ (t_x,t_y) $$, in any order, would be

$$

\left( \begin{array}{c} x_{in} \\ y_{in} \end{array} \right) =

\left[ \begin{array}{cc} s \ cos \ \alpha & - s \ sin \ \alpha & t_x \\ s \ sin \ \alpha & s \ cos \ \alpha & t_y  \end{array} \right]

\left( \begin{array}{c} x_{out} \\ y_{out} \\ 1 \end{array} \right)

$$

Now we have all the maths !

The second important thing about SPN is that they are **trainable** : to learn the parameters to predict the transformation, SPN can **retropropagate gradients**.



# Spatial Transformer Networks in Caffe

I updated [Caffe with Carey Mo implementation](https://github.com/christopher5106/last_caffe_with_stn) :

    git clone https://github.com/christopher5106/last_caffe_with_stn.git

Compile Caffe following my tutorial on [iOS](http://christopher5106.github.io/big/data/2015/07/16/deep-learning-install-caffe-cudnn-cuda-for-digits-python-on-mac-osx.html) or [Ubuntu](http://christopher5106.github.io/big/data/2015/07/16/deep-learning-install-caffe-cudnn-cuda-for-digits-python-on-ubuntu-14-04.html).

Let's create our first SPN to see how it works :

```

name: "stn"
input: "data"
input_shape {
  dim: 1
  dim: 3
  dim: 227
  dim: 227
}
input: "theta"
input_shape {
  dim: 1
  dim: 2
}
layer {
  name: "st_1"
  type: "SpatialTransformer"
  bottom: "data"
  bottom: "theta"
  top: "transformed"
  st_param {
    to_compute_dU: false
    theta_1_1: 0.5
    theta_1_2: 0.0
    theta_2_1: 0.0
    theta_2_2: 0.5
  }
}

```

```

name: "stn"

layer {
  name: "data"
  type: "ImageData"
  top: "data"
  top: "label"
  image_data_param {
    source: "results.csv"
    batch_size: 1
  }
}
input: "theta"
input_shape {
  dim: 1
  dim: 2
}

layer {
  name: "st_1"
  type: "SpatialTransformer"
  bottom: "data"
  bottom: "theta"
  top: "transformed"
  st_param {
    to_compute_dU: false
    output_H: 224
    output_W: 224
    theta_1_1: 0.5
    theta_1_2: 0
    theta_2_1: 0
    theta_2_2: 0.5
  }
}

```
