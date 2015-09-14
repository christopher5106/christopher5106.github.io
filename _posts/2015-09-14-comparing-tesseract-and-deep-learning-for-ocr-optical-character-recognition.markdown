---
layout: post
title:  "Compare Tesseract and deep learning techniques for Optical Character Recognition of licence plates"
date:   2015-09-14 23:00:51
categories: computer vision
---

![Cat]({{ site.url }}/img/plate_recognition.jpg)

I first created a simple "plate annotation tool"

    ./annotate input_dir output.csv

in order to create a txt file labelling the data, one line per character

    image_path  character  x  y width height orientation

in a CSV format. Licence plates are detected with a cascade classifier and letters with the findContours method from OpenCV.

I created a convertion tool

    ./extract file.csv output_dir --backend=[lmdb|leveldb|directory|tesseract]

to convert this file to format for [Tesseract]({{ site.url }}/optical/character/recognition/2015/09/01/training-optical-character-recognition-technology-tesseract.html) and [Caffe learning]({{ site.url }}/deep/learning/2015/09/04/Deep-learning-tutorial-on-Caffe-Technology.html).

For Tesseract, this will bring me such a file :

![Tiff file for Tessearct]({{ site.url }}/img/lpfra.std.exp0.jpg)

For Caffe, this will populate a LMDB database that I can inspect in Python :

{% highlight python %}
import caffe
import numpy as np
import lmdb
import matplotlib.pyplot as plt

env = lmdb.open('test_lmdb')
t = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','0','1','2','3','4','5','6','7','8','9'];

def get(k):
    with env.begin() as txn:
        raw_datum = txn.get(k)

    datum = caffe.proto.caffe_pb2.Datum()
    datum.ParseFromString(raw_datum)
    flat_x = np.fromstring(datum.data, dtype=np.uint8)
    x = flat_x.reshape(datum.channels, datum.height, datum.width)
    y = datum.label
    plt.imshow(x[0,...],cmap='gray')
    plt.show()
    print t[y]

get(b'00000006')
get(b'00000009')
{% endhighlight %}

The training set is composed of 5000 letters, and the test set of 160 letters.

I used a standard LeNet neural network with dropout layers.

I trained both technologies and here is the result :

| Technology        | Correct results           |
| ------------- |:-------------:|
| Tesseract eng language      | 72 |
| Tesseract trained      | -      |
| Caffe trained (NN)  | 154 |

Caffe is 97% right. The wrong matches are :

M W
0 D
B 8
1 A
D 0
D Q

Given that we know infer the letter/number shema for a licence plate (LL-NNN-LL or NN-LLL-NN) with a good precision, it's in fact a 99% correctness, that means one wrong letter every hundred letters with Caffe.