---
layout: post
title:  "Object detection deep learning frameworks for Optical Character Recognition and Document Pretrained Features"
date:   2018-06-26 00:00:00
categories: deep learning
---

Working as AI architect at Ivalua company, I'm happy to announce the release in the open source of my code for optical character recognition using Object Detection deep learning techniques.

The main purpose of this work was to compute pretrained features that could serve as early layers in more complex deep learning nets for document analysis, segmentation, classification or reading.

Object detection task has been improving a lot with the arise of new deep learning models such as R-CNN, Fast-RCNN, Faster-RCNN, Mask-RCNN, Yolo, SSD, ... These models have been developped on the case of natural images, on datasets such as COCO, Pascal VOC, ... They have been applied to the task of digit recognition in natural images, such as the Street View House Numbers (SVHN) Dataset.

These object detection models have sometimes been applied to documents at the global document scale to extract document zones, such as tables or document layouts. But documents are very different from natural images, mainly black and white, with very strong image gradients and very different gradient patterns. Object detection models are based on pretrained features coming from classification networks on natural images, such as AlexNet, VGG, ResNets, ... and might not be suited to document, in particular at character level, to recognize characters, and it might have a negative impact at the global document level.

The classification network basis for document images could certainly be better found in networks developped for the MNIST digit dataset, such as the LeNet. The idea is to explore the application of object detection networks to character recognition with network architectural backbones inspired by these digit classification networks, better suited to document image data.

Fitting the full document image is quite challenging, since characters could be hardly read under a document size lower than 1000 pixel high, and classically, training deep learning networks for image tasks is usually performed on small sized images, less than 300 pixels high and wide (224, 256, ...).

The code has been tested on toy examples, built with MNIST data:

<img src="img/ocr/res1.png" height="250"> <img src="img/ocr/res2.png" height="250">

Then, as classically done in object detection, the code has been evaluated on image crops using multiple layers as in SSD to recognize characters at different scale:

<img src="img/ocr/res3.png" height="250"> <img src="img/ocr/res4.png" height="250">

Last, a low resolution and a small batch size have permitted to fit the image and the network in the GPU, and the following result has been achieved, droping the characters of size too big or too small:

<img src="img/ocr/res5.png" height="250">

The full expirement settings and results are described in the [PDF paper](img/ocr/Object_detection_deep_learning_networks_for_Optical_Character_Recognition.pdf).

We hope that researchers and the open source community will follow up on our work to invent more accurate or more efficient networks for full document processing, that could serve as early layers for further document tasks.

**Well done!**
