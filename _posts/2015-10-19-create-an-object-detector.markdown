---
layout: post
title:  "Create an object detector with OpenCV Cascade Classifier : best practice and tutorial"
date:   2015-10-19 23:00:51
categories: computer vision
---

To have a quick start, try this [example](https://github.com/mrnugget/opencv-haar-classifier-training)

    git clone https://github.com/mrnugget/opencv-haar-classifier-training.git

In my example, I will train the classifier with training windows of size

    WIDTH=100
    HEIGHT=20

The dimensions specify the smallest object size the classifier will be able to detect. Objects larger than that will be detected by the multiscale image pyramid approach.

## Extracting rectangles to OpenCV format

As a best practice, I would recommend to create an executable, `extract`, to extract training windows, positive ones as well as negative ones, from an annotated input of your choice :

    ./extract input.csv $WIDTH $HEIGHT


The purpose of my `extract` program is to create two directories that can be directly used by OpenCV cascade training algorithm :

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

while `pos/info.dat` contains rectangle informations

    img/xxxx.png 1 x y w h
    img/yyyy.png 1 x y w h

In the `pos/img/` images are full size, since the rectangle information is in the `info.dat` file, whereas the `neg/img/` images are extracted.

In my case I provide many more negatives than positives to the classifier  (4 times more).

I would avoid leaving OpenCV training algorithm create the negative windows (`opencv_traincascade` subsample negative image), or to do that, my `extract` will create the background images at the final training size (100x20 in my example) so that it cannot subsample but only take the entire negative image as a negative.

Creating negatives from the backgrounds of the positives is much more "natural" and will give far better results, than using a wild list of background images taken from the Internet. That's all that makes the interest of such an `extract` program.



The CSV input file to the program is a list of input images with the class and coordinates of the rectangles where objects are located in the image,

    /Users/christopherbourez/data/img.png,0,10,30,210,65

The last two input parameters give the size to resize the negative windows after extraction.


## OpenCV positives preprocessing

It is required to use an OpenCV program to convert the positive rectangles to a new required format :

    opencv_createsamples -info pos/info.dat -vec pos.vec -w $WIDTH -h $HEIGHT

You could also augment the positive sample by rotating and distorting the images with `opencv_createsamples` and merging them back into one vec with Naotoshi Seo’s `mergevec.cpp` tool.


## Train the classifier

    NUMPOS=1000
    RATIO=2

    mkdir models

    opencv_traincascade -data models -vec pos.vec -bg neg/info.dat -w $WIDTH -h $HEIGHT -numPos $(expr $NUMPOSTRAIN*0.85/1 |bc) -numNeg $(expr $RATIO*$NUMPOSTRAIN*0.85/1 |bc)  -precalcValBufSize 1024 -precalcIdxBufSize 1024

    # or

    opencv_traincascade -data models -vec pos.vec -bg neg/info.dat -w $WIDTH -h $HEIGHT -numStages 20 -minHitRate 0.999 -maxFalseAlarmRate 0.5 -numPos $(expr $NUMPOSTRAIN*0.85/1 |bc) -numNeg $(expr $RATIO*$NUMPOSTRAIN*0.85/1 |bc) -mode ALL -precalcValBufSize 1024 -precalcIdxBufSize 1024




About the training parameters :

- `numPos` parameter has to be about 90% of the number of positive rectangles, since some positives that are too different from the the positive set can be rejected by the algorithm and if `numPos` equals the number of positives, it will fail with the following message :

        OpenCV Error: Bad argument (Can not get new positive sample. The most possible reason is insufficient count of samples in given vec-file.

    Increasing the number of positives will enable a better generalization of the model. Usually a few thousand is good.

- `numNeg` : it is usually good to take two times more negatives than positives.

    Increasing the number of negative will diminish the number of false positive detections.

- increasing `numStages` will not improve anymore the model when overfitting occurs. In this case, you'll need to add more positives and negatives to the training set.

Be careful also, the JS library `jsfeat` only accept detectors in the old format (use `opencv_haartraining` instead).


## Use in nodeJS

Simply create a **recognize.js** program :

{% highlight javascript %}
var cv = require("opencv");

var color = [0, 255, 0];
var thickness = 2;

var cascadeFile = "models/cascade.xml";

var inputFiles = [ "image.jpg" ];

inputFiles.forEach(function(fileName) {
  cv.readImage(fileName, function(err, im) {
    im.detectObject(cascadeFile, {neighbors: 2, scale: 2}, function(err, objects) {
      console.log(objects);
      for(var k = 0; k < objects.length; k++) {
        var object = objects[k];
        im.rectangle([object.x, object.y], [object.width, object.height], color, 2);
      }
      im.save(fileName.replace(/.jpg/, "processed.jpg"));
    });
  });
});
{% endhighlight %}

and call the detector

    node recognize.js

**Well done !**

##more

A few posts : [1](http://coding-robin.de/2013/07/22/train-your-own-opencv-haar-classifier.html)
- [2](http://note.sonots.com/SciSoftware/haartraining.html)
- [3](http://opencvuser.blogspot.be/2011/08/creating-haar-cascade-classifier-aka.html)
