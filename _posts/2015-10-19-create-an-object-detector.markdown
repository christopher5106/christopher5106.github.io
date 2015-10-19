---
layout: post
title:  "Create an object detector with OpenCV Cascade Classifier : best practice"
date:   2015-10-19 23:00:51
categories: computer vision
---

In order to gain information, read a few posts : [1](http://coding-robin.de/2013/07/22/train-your-own-opencv-haar-classifier.html)
- [2](http://note.sonots.com/SciSoftware/haartraining.html)
- [3](http://opencvuser.blogspot.be/2011/08/creating-haar-cascade-classifier-aka.html)


To have a quick start, try this [example](https://github.com/mrnugget/opencv-haar-classifier-training)

    git clone https://github.com/mrnugget/opencv-haar-classifier-training.git


As a best practice, I would recommend to create an executable `extract` to create training windows, positive as well as negative ones :

    ./extract input.csv 100 20

The CSV file contains the list of images with the coordinates of the rectangles where objects are located, and the last two parameters correspond to the size to resize the windows after extraction.

I would avoid to leave the creation of negative windows to the `opencv_traincascade` program, and to use a wild list of background images : I prefer to extract my own background images from the images where the objects have been found, because they are more realistic backgrounds for these objects. In order to have `opencv_traincascade` program use my windows as negative windows, `extract` will create the background images at the final training size (100x20 in my example).

That's why the `extract` program create two directories

    pos
    -- info.dat
    -- img
    ---- xxx.png
    ---- yyyy.png
    neg
    -- info.dat
    -- img
    ---- zzzz.png
    ---- llll.png

`neg/info.dat` is a simple list of images

    img/zzzz.png
    img/llll.png

while `pos/info.dat` contains also rectangle informations

    img/xxxx.png 1 x y w h
    img/yyyy.png 1 x y w h

In the `pos/img/` images are full size, since the rectangle information is in the `info.dat` file, whereas the `neg/img/` images are extracted.

In my case I provide many more negatives than positives to the classifier  (4 times more).


Then to extract the positive rectangles :

    opencv_createsamples -info pos/info.dat -vec pos.vec -w 100 -h 20

and to train the classifier :

    mkdir models
    opencv_traincascade -data models -vec pos.vec -bg neg/info.dat -w 100 -h 20 -nstages 20 -nsplits 2 -minhitrate 0.999 -maxfalsealarm 0.5 -numPos 1000 -numNeg 2000 -mem 2048 -mode ALL
