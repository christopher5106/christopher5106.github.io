---
layout: post
title:  "ROI max pooling and shareable layers : fast and accurate deep learning nets."
date:   2015-09-23 23:00:51
categories: computer vision
---

As seen in [previous post](http://christopher5106.github.io/computer/vision/2015/09/15/deep-learning-net-surgery-to-create-a-feature-map.html), I got a classification feature map

![Feature map]({{ site.url }}/img/caffe_immat_feature_map.png)

Let's go further to get precise position information about the license plate and its letters.

I will re-use the first 2 convolution layers to create a feature map over which I will slide a window of 3x3 on top of which will operate two new nets :

- a box classification layer, giving the probability of the licence plate to be centered on this point

- a box regression layer, giving the size of the box on that point

A sliding 'inner product net' on a window of 3x3 can be done with a simple 'convolution net' of kernel 3.

At the end, the effective receptive field on the input image will be of size XxX.
