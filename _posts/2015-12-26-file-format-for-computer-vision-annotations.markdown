---
layout: post
title:  "Which file format for computer vision annotations?"
date:   2015-12-26 23:00:51
categories: computer vision
---

This is a good question when you start a computer vision project. You are going to annotate your pictures in a format that will serve different computer vision tools for training.

In the [article about creating an object dectector with OpenCV traincascade algorithm](http://christopher5106.github.io/computer/vision/2015/10/19/create-an-object-detector.html), we saw the OpenCV format that has main drawbacks :

- no orientated rectangles : some real world situations require to take orientation into consideration

- no class : OpenCV format is designed to train only one category of detector

- separation of "positives" and "negatives", that are not a classification.

It would be tempting to create CSV files in the OpenCV Rect format, with a top left corner coordinates :

    path,class,x,y,w,h

To my point of view, the best format remains the **RotatedRect format** with center point coordinates, and orientation :

    path,class,center_x,center_y,w,h,o

It will be a more general case, orientation is very common, and this will enable you to use the same tools to work on your data, extract rectangles, etc. In this format, it is also quite easy to work without the orientation by ignoring the last column, if wanted.

It is also possible to go further by adding perspective information and rotation on other axis, but in any case, center position is better than top left corner position for such description.

I will also give 2 other precious advices :

- coordinates have to be written for the **original-sized image** (not a resized image), because sometimes, you will have to extract parts of the image and read some data in them, and full size image will give a better recognition quality than resized images.

- save the CSV file either in the image directory, or next to your image directory so that image path can be relative (and not absolute), to work on different computers. The second solution, **next to the image directory**, is better to include in the same CSV file two different image folders, very common in computer vision.
