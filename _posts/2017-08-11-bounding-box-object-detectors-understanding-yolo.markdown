---
layout: post
title:  "Bounding box object detectors: understanding YOLO"
date:   2017-08-10 00:00:51
categories: object detectors
---

In this article, I re-explain the characteristics of the bounding box object detector Yolo since everything might not be so easy to catch. Its [first version](https://pjreddie.com/media/files/papers/yolo.pdf) has been improved in a [version 2](https://arxiv.org/pdf/1612.08242.pdf).


# Grid

The convolutions enable to compute predictions at different positions in an image in an optimized way. This avoids using a *sliding window* to compute separately a prediction at every potential position.

The deep learning network for prediction is composed of convolution layers of stride 1, and max-pooling layers of stride 2 and kernel size 2.

To make it easier to align the grid of predictions produced by the network with the image:

- each  max-pooling layer divides the output size by 2, multiplies the network stride by 2, and shifts the position in the image corresponding to the net receptive field center by 1 pixel. The padding mode used is 'VALID', meaning it is the **floor division** in the image size division, the extra values left on the right and the bottom of the image will only be used in the network field but a smaller image without these pixels (as the blue one in the figure below) would lead to the same grid;

- the padding mode 'SAME' used in the convolutions means the output has the same size as the input. This is performed by padding with some zeros (or other new values) each border of the image. If we'd used a 'VALID' padding for the convolutions, then the first position on the grid would be shifted by half the network reception field size, which can be huge (~ 200 pixels for a ~400 pixel large network). The problem requires us to predict objects close to the borders, so, to avoid a shift and start predicting values at the image border, borders are padded by half the reception field size, an operation performed by the 'SAME' padding mode.

The following figure displays the impact of the padding modes and the network outputs in a grid. Final stride is $$ 2^{\text{nb max-pooling layers}} $$, and the left and top offsets are half that value, ie $$ 2^{\text{nb max-pooling layers - 1}} $$:

![]({{ site.url }}/img/yolo-padding.png)


# Positives and negatives

A position on the grid, that is the closest position to the center of one of the ground truth bounding boxes, is **positive**. Other positions are negative.

![]({{ site.url }}/img/yolo-positives.png)

So, let us keep theses circles as a representation of the net outputs, and for the grid,  rather than displaying a grid of the outputs, let us use it to display the cells for which any ground truth box center  will make these positions as positive and shift it back by half a stride:

![]({{ site.url }}/img/yolo-grid.png)


# A regressor rather than a classifier

For every positive position, the network predicts a **regression** on the bounding box precise position and dimension.

In the second version of Yolo, the predictions are relative to the grid position and anchor size (instead of the full image) as in the Faster-RCNN models for better performance:

$$ b_x = \sigma(t_x) + c_x $$


$$ b_y = \sigma(t_y) + c_y $$


$$ b_w = p_w \ e^{t_w} $$


$$ b_h = p_h \ e^{t_h} $$

where $$ (c_x, c_y) $$ are the grid cell coordinates and $$ (p_w, p_h) $$ the anchor dimensions.

![]({{ site.url }}/img/yolo-regression.png)


# Confidence

Once the bounding box regressor is trained, the model is trained to predict a confidence score on the final predicted bounding box with above regression.

The natural confidence score value is:

- for positive positions, the **intersection over union (IOU)** of the predicted box with the ground truth

- for negative positions, zero.

In the Yolo papers, confidence is trained jointly with the position/dimension regressor, which can cause model instability. To avoid this, they weighted the regressor loss 5 times the confidence regressor.


# Anchors or prediction specialization

Yolo V1 and V2 predict regression for B bounding boxes. Only one of the B regressors is trained at each positive position, the one that predicts a box that is closest to the ground truth box, so that there is a reinforcement of this predictor, and a specialization of each regressor.

In Yolo V2, this specialization is 'assisted' with predefined anchors as in Faster-RCNN. The predefined anchors are chosen as representative as possible of the ground truth boxes, with a K-means clustering to find them.

# All together

For each specialization, in Yolo V2, the class probabilities of the object inside the box is trained to be predicted, as the confidence score.

Putting it all together for an example of 98 anchors, 10 object classes, the output of the network at each position can be decomposed into 3 parts:

![]({{ site.url }}/img/net_output.png)

For all outputs except the width and height scaling, the outputs are (logistic activation function or sigmoid)

**Well done!**
