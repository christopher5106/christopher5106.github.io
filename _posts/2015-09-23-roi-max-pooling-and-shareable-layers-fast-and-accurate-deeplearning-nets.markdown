---
layout: post
title:  "ROI max pooling and shareable layers : fast and accurate deep learning nets."
date:   2015-09-23 23:00:51
categories: computer vision
---


As seen in [previous post](http://christopher5106.github.io/computer/vision/2015/09/14/comparing-tesseract-and-deep-learning-for-ocr-optical-character-recognition.html), deep nets read the correct letter with a correctness of 99%.

Now let's go further to get precise position information about the license plate and its letters, as explained in [Faster RCNN publication](https://github.com/ShaoqingRen/faster_rcnn) from Microsoft Research two weeks ago.

I will re-use the first 2 convolution layers to create a feature map over which I will slide a window of 3x3 on top of which will operate two new nets :

- a box classification layer, giving the probability of the licence plate to be centered on this point

- a box regression layer, giving the size of the box on that point

These two new nets are composed of a common innerproduct layer "ip1-rpn", and another innerproduct layer specific to each of them.

Instead of training theses nets on the feature map, I will train the full net composed of the first 2 convolution layers and the two new nets, but with a learning rate set to 0 for the first 2 layers. At the end, the effective receptive field on the input image will be of size XxX.

During testing and deployment, the sliding 'inner product net' on a window of 3x3  will be replaced with a simple 'convolution net' of kernel 3 with the same parameters.

A a dataset, I labeled the letters on each images and I can easily extract the plate zone. Statistics of license plates are :


|        | Average          | Max          | Min          |
| ------------- |:-------------:|
| Width | 200 | 673 | 61 |
| Height | 40 | 127 | 15 |
| Min orientation | - | 25 |  -25 |

<br/>

So I will consider the output of the nets to predict the probability and regression of 5 anchors at 5 different scales / widths : 660x330 - 560x187 - 460x153 - 360x120 - 260x87 - 160x53 - 60x20.

#Train net

Here are the different steps for the train net :

1. re-use the previous net parameters for the shared layer that will have the same names : conv1, pool1, conv2, pool2.

2. fix their learning rate at 0 :

        param {
          lr_mult: 0
          decay_mult: 0
        }

3. keep the dropout layer after the convolution layers

4. change the name of the innerproduct layer "ip1" for "ip1-rpn" to train with new random weight params and replace the innerproduct layer "ip2" with 2 sibling convolutional layers :

    - "cls_score" with 5 x 2 parameters (the probability or not to be a plate)

    - "bbox_pred" with 5 x 4 parameters : t_x, t_y, t_w (width) and t_o (orientation)

5. add a "SoftmaxWithLoss" layer for cls_score and "SmoothL1Loss" layer for bounding box regression layer.

6. data layer

I will feed the data layer with extracted rectangles images, and for each reactangle, a label, position t_x, t_y, t_w and t_o.


#Feature map net

Let's train with the previously learned parameters the new model :

    ~/technologies/caffe/build/tools/caffe train --solver=lenet_train_test_position.prototxt -weights=lenet_iter_2000.caffemodel -gpu 0

Once trained, I convert the innerproduct layers into the convolution layers to get a feature map, as seen in [previous post](http://christopher5106.github.io/computer/vision/2015/09/15/deep-learning-net-surgery-to-create-a-feature-map.html).

![Feature map]({{ site.url }}/img/caffe_immat_feature_map.png)


#Test/deploy net

On top of the feature map layer, add a NMS layer and a Top-N layer and a ROI pooling layer at the place of the dropout layer :

    layer {
      name: "roi_pool3"
      type: "ROIPooling"
      bottom: "pool2"
      bottom: "rois"
      top: "pool3"
      roi_pooling_param {
        pooled_w: 7
        pooled_h: 7
        spatial_scale: 0.0625 # 1/16
      }
    }

Creating our own NMS and Top-N layer.

###Note : Caffe in C++

The **blob** ([blob.hpp](https://github.com/BVLC/caffe/blob/master/include/caffe/blob.hpp) and [blob.cpp](https://github.com/BVLC/caffe/blob/master/src/caffe/blob.cpp)) is a wrapper to manage memory independently of CPU/GPU choice, using [SyncedMemory class](https://github.com/BVLC/caffe/blob/master/src/caffe/syncedmem.cpp), and has a few functions like Arrays in Python, both for the data and the computed gradient (diff) arrays contained in the blob :

- shape() and shape_string() to get the shape, or shape(i) to get the size of the i-th dimension, or shapeEquals() to compare shape equality
- reshape() or reshapeLike() another blob
- offset() to get the c++ index in the array
- CopyFrom() to copy the blob
- data_at() and diff_at()
- asum_data() and asum_diff() their L1 norm
- sumsq_data() and sumsq_diff() their L1 norm
- scale_data() and scale_diff() to multilply the data by a factor

A layer, such as the [SoftmaxWithLoss layer](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/softmax_loss_layer.cpp), will need a few functions working with arguments top blobs and bottom blobs :

- Forward_cpu or Forward_gpu
- Backward_cpu or Backward_gpu
- Reshape
- optionaly : LayerSetUp, to set non-standard fields

The [layer_factory](https://github.com/BVLC/caffe/blob/master/src/caffe/layer_factory.cpp) is a set of helper functions to get the right layer implementation according to the engine (Caffe or CUDNN).

![png]({{ site.url }}/img/cars.png)
