---
layout: post
title:  "Supervised learning, unsupervised learning with Spatial Transformer Networks tutorial in Caffe and Tensorflow : let's see a document classifier"
date:   2016-02-29 23:00:51
categories: big data
---


# Spatial Transformer Networks

Spatial Transformer Networks is a hot-off-the-press network invented by [Max Jaderberg, Karen Simonyan, Andrew Zisserman, Koray Kavukcuoglu](http://arxiv.org/pdf/1506.02025v3.pdf) at Google DeepMind artifical intelligence lab in London.

Spatial transformer networks perform an affine transformation given :

![]({{ site.url }}/img/spatial_transformer_networks.png)

$$

\left( \begin{array}{c} x_{in} \\ y_{in} \end{array} \right) =

\left[ \begin{array}{cc} \theta_{11} & \theta_{12} & \theta_{13} \\ \theta_{21} & \theta_{22} & \theta_{23} \end{array} \right]

\left( \begin{array}{c} x_{out} \\ y_{out} \\ 1 \end{array} \right)

$$

The $$ (x,y) $$ are normalized coordinates, $$ (x,y) \in [-1,1] \times [-1,1]  $$ .

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

**Now we have all the maths !**

# Spatial Transformer Networks in Caffe

Let's give a try with [Carey Mo implementation](https://github.com/daerduoCarey/SpatialTransformerLayer).

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

# Supervised learning

Let's extract windows for supervised learning. I'll extract negative windows, that do not contain the object, and positive windows that will contain the object.

Once ground truth has been annotated with rectangles, conceptually, there are two main factors :

- **the maximum scale factor**, which corresponds to the maximal value for the *s* variable in our spatial transformer network, but during the process of generating positive windows, will give you by which factor the positive window can be relatively to the annotation rectangle.

- **the negative window scale**, which corresponds to the ratio of the *neural network field width* with the image width at which you want to perform your detection. This parameters gives you the width of your negative window during generation of negative windows.

For example, if I use the flags *--resize_width=227 --ratio=1.0 --noise_scale=3 --neg_width=0.3* to extract my windows, I expect :

- my network to have a field of 227x227 (height is given by width x ratio)

- images to be resized before detection to a width of 227 / 0.3 = 756.

- objects to have a width between 75 (227 / 3) to 227 pixels.

We usually 4 to 5 more negatives per positive.

# Unsupervised learning
