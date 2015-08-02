---
layout: post
title:  "GPU accelerated computing versus cluster computing for machine / deep learning"
date:   2015-07-31 23:00:51
categories: big data
---

Microsoft Research in 2013 released this article that [nobody got fired for buying a cluster](http://research.microsoft.com/pubs/179615/msrtr-2013-2.pdf). At that time, optimizations on CPU were already a very interesting point in computation.

Nowadays, it's even more the case with GPU :

![Google Brain]({{ site.url }}/img/deeplearning/google_brain_versus_three_gpu.png)

The new approach of deep learning:

![Deep learning]({{ site.url }}/img/deeplearning/deeplearning.png)

Practical examples from NVIDIA :

![Deep learning]({{ site.url }}/img/deeplearning/examples.png)

The traditional approach of feature engineering :

![Deep learning]({{ site.url }}/img/deeplearning/traditionnal.png)

where the main problem was to find the correct definition of features.

And the new deep learning approach :

![Deep learning]({{ site.url }}/img/deeplearning/approach.png)

is inspired by nature :

![Deep learning]({{ site.url }}/img/deeplearning/neurons.png)

with the following advantages :

![Deep learning]({{ site.url }}/img/deeplearning/advantages.png)


Here are my installs of Berkeley's deep learning library [Caffe](http://caffe.berkeleyvision.org/) and NVIDIA deep learning interactive interface [DIGITS](https://developer.nvidia.com/digits) on NVIDIA GPU :

- on [iMac]({{ site.url }}/big/data/2015/07/16/deep-learning-install-caffe-cudnn-cuda-for-digits-python-on-mac-osx.html)

- on [AWS g2 instances]({{ site.url }}/big/data/2015/07/16/deep-learning-install-caffe-cudnn-cuda-for-digits-python-on-ubuntu-14-04.html)

Installs on mobile phones :

- [Android](https://github.com/sh1r0/caffe-android-lib)

- [iOS](https://github.com/noradaiko/caffe-ios-sample)

Clusters remain very interesting for parsing and manipulating large files such as for example [parsing Wikipedia pages with Spark]({{ site.url }}/bigdata/2015/05/28/parse-wikipedia-statistics-and-pages-with-Spark.html).
