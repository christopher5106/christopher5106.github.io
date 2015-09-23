---
layout: post
title:  "ROI max pooling and shareable layers : fast and accurate deep learning nets."
date:   2015-09-23 23:00:51
categories: computer vision
---

As seen in [previous post](http://christopher5106.github.io/computer/vision/2015/09/15/deep-learning-net-surgery-to-create-a-feature-map.html), I got a classification feature map

![Feature map]({{ site.url }}/img/caffe_immat_feature_map.png)

Let's go further to get position information about the license plate and its letters.

I will re-use the first 2 convolution layers to create a feature map over which I will slide two new nets over a window of 3x3 :

- a box classification layer, giving the probability of the licence plate to be centered on this point

- a box regression layer, giving the size of the box on that layer

These sliding nets can be done via convolution layers of kernel 3 on top of the previous net.

At the end, the effective receptive field on the input image will be of size XxX.
