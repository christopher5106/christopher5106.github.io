---
layout: post
title:  "Bounding box object detectors: understanding YOLO"
date:   2017-08-10 00:00:51
categories: object detectors
---

In this article, I re-explain the characteristics of the bounding box object detector Yolo since everything might not be so easy to catch. Its [first version](https://pjreddie.com/media/files/papers/yolo.pdf) has been improved in a [version 2](https://arxiv.org/pdf/1612.08242.pdf).


# Grid

The convolutions enable to compute a detection score at different positions in an optimized way. This avoids using a *sliding window* to compute a score at every potential position.

The network is composed of convolution layers of stride 1, and max-pooling layers of stride 2 and kernel size 2.

This way, it is easier to align the grid produced by the network with the image:

- each  max-pooling layer divide the image size by two and multiply the stride by two. The padding used is 'VALID', meaning it is the **floor division** in the image size division, the extra values left on the right and the bottom of the image will only be used in the network field but a smaller image without these pixels would lead to the same grid

- the convolutions have the 'SAME' padding, meaning the output has the same size of the output, which is performed by adding some zeros (or other values) on each border of the image. If we use a 'VALID' padding for the convolutions, then the first position on the grid would be shifted by half the network reception field size, which can be huge (~ 200 pixels for a ~400 pixel large network). The problem requires us to predict objects close to the borders also, so to avoid that shift, borders are padded by half the reception field size, which is performed by the 'SAME' padding.

![]({{ site.url }}/img/yolo-padding.png)


# Positives and negatives

Positions on the grid belong to a cell in which one of ground truth bounding boxes has a center inside are **positive**. Other positions are negative.

![]({{ site.url }}/img/yolo-positives.png)


# A regressor rather than a classifier

For every positive position, the network predicts a **regression** on the bounding box precise position and dimension.

In the second version of Yolo, the predictions are relative to the grid position and anchor size (instead of the full image) as in the Faster-RCNN models:

$$ b_x = \sigma(t_x) + c_x $$
$$ b_y = \sigma(t_y) + c_y $$
$$ b_w = p_w \ e^{t_w} $$
$$ b_h = p_h \ e^{t_h} $$

where $$ (c_x, c_y) $$ is the grid cell coordinate and $$ (p_w, p_h) $$ the anchor dimension.

![]({{ site.url }}/img/yolo-regression.png)


# Confidence

Once the bounding box regressor is trained, the model is trained to predict a confidence score on the predicted bounding box with above parameters.

The natural confidence score value is:

- for positive positions, the **intersection over union (IOU)** of the predicted box with the ground truth

- for negative positions, zero.

In the Yolo papers, confidence is trained jointly with the position/dimension regressor, which can cause model instability. To avoid this, they weighted the regressor loss 5 times the confidence regressor.


# Anchors or prediction specialization

Yolo V1 and V2 predicts regression for B bounding boxes. The regression is trained at each positive position only for the predicted boxes that is closest to the ground box, so that there is a reinforcement of the predictor.

In Yolo V2, this specialization is 'assisted' with predefined anchors as in Faster-RCNN. The predefined anchors are choosen as representative as possible of the ground truth boxes, with a K-means to compute them.

# All together

For each specialization, in Yolo V2, a class of the object in the box is predicted as well as the confidence.

Putting all together on for an example of 10 classes and 98 anchors, the prediction of the network at each position can be decomposed into 3 parts:

![](img/net_output.png)

**Well done!**
