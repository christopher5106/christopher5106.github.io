---
layout: post
title:  "OpenCV train cascade versus Caffe deep neural networks for object localization"
date:   2015-11-06 23:00:51
categories: computer vision
---

Let's take the CaffeNet

Cut the 3 innerproduct layers
Replace them with 2 new innerproduct layers
that we initiate with weight_filler and bias_filler

Set learning rate to zero for parameters of inferior levels and
load previous parameters


Transform the results into a convolution

    python transform_to_convolution.py cls.prototxt cls.caffemodel map.prototxt map.caffemodel


    python transform_to_convolution.py  /Users/christopherbourez/technologies/caffe/models/bvlc_reference_caffenet/deploy.prototxt /Users/christopherbourez/technologies/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel bvlc_reference_caffenet-conv.prototxt  bvlc_reference_caffenet-conv.caffemodel
