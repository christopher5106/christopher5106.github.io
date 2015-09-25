---
layout: post
title:  "ROI max pooling and shareable layers : fast and accurate deep learning nets."
date:   2015-09-23 23:00:51
categories: computer vision
---


As seen in [previous post](http://christopher5106.github.io/computer/vision/2015/09/14/comparing-tesseract-and-deep-learning-for-ocr-optical-character-recognition.html), deep nets read the correct letter with a correctness of 99%.

Now let's further to get precise position information about the license plate and its letters.

I will re-use the first 2 convolution layers to create a feature map over which I will slide a window of 3x3 on top of which will operate two new nets :

- a box classification layer, giving the probability of the licence plate to be centered on this point

- a box regression layer, giving the size of the box on that point

A sliding 'inner product net' on a window of 3x3 can be done with a simple 'convolution net' of kernel 3.

At the end, the effective receptive field on the input image will be of size XxX.

I labeled the letters and I can easily extract the plates. Statistics of license plates are :

Average rectangle width : 200
Average rectangle height : 40

Max width : 673
Min width : 61

Max height : 127
Min height : 15

Min orientation : -25
Max orientation : 25

So I will consider the output of the nets to predict the probability and regression of 5 anchors at 5 different scales : width of 660 - 560 - 460 - 360 - 260 - 160 - 60.

#Train net

Here are the different steps for the train net :

1. re-use the previous net parameters, so common layer will have the same names : conv1, pool1, conv2, pool2.

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

I will feed the data layer with extracted rectangles images, a label, position t_x, t_y, t_w and t_o.

#Feature map net

Let's train with the previously learned parameters the new model :

    ~/technologies/caffe/build/tools/caffe train --solver=lenet_train_test_position.prototxt -weights=lenet_iter_2000.caffemodel -gpu 0

Once trained, we can convert the innerproduct layers into the convolution layers to get a feature map, as seen in [previous post](http://christopher5106.github.io/computer/vision/2015/09/15/deep-learning-net-surgery-to-create-a-feature-map.html).

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
