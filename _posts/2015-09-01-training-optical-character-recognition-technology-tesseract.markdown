---
layout: post
title:  "Training optical character recognition technology Tesseract on a new character font on MacOS"
date:   2015-09-01 23:00:51
categories: optical character recognition
---

#Training Tesseract on a new font

First install Tesseract

    brew install tesseract

Let's create a new language "newfra" :

{% highlight bash %}
newlang="newfra"
{% endhighlight %}

Tesseract will use a **TIFF image file** (with characters to learn) and a **Box file** (indicating the bounding box of the characters in the image) to do its training to a new language.

First begin by creating the character table as a TIFF image. Here is an example of TIFF file :

![TIFF File](https://printalert.files.wordpress.com/2014/04/ocr_input.jpg)

The format of the box file is one ligne per character in the image and each line of the form `char bl_x bl_y rt_x rt_y` where `char` is the character, `bl_x` the abcyss of bottom-left corner in a coordinate system where (0,0) is at the bottom-left corner of the TIFF image.

To create the box file, it's possible to use Tesseract recognition engine and manually add/complete the lines that were not recognized automatically, correct lines that were recognized improperly.

{% highlight bash %}
tesseract ${newlang}.std.exp0.tif ${newlang}.std.exp0 -l fra batch.nochop makebox # '-l fra' is optional, if use of the French language recognition
{% endhighlight %}

To perform a better recognition, you can download additional languages such as "fra" for French language. Put the file `fra.traineddata` in `/usr/local/share/tessdata/` for Tesseract to use it. You can also use an [online tool](http://pp19dd.com/tesseract-ocr-chopper/).

Then, with the 2 files (.tiff and .box), here is the list of commands to create a new language `newfra` for Tesseract from this TIFF image :

{% highlight bash %}
tesseract ${newlang}.std.exp0.tif ${newlang}.std.exp0 box.train.stderr
unicharset_extractor ${newlang}.std.exp0.box
echo "std 0 0 0 0 0" > font_properties
shapeclustering -F font_properties -U unicharset ${newlang}.std.exp0.tr
mftraining -F font_properties -U unicharset -O lfra.unicharset ${newlang}.std.exp0.tr
cntraining ${newlang}.std.exp0.tr
mv normproto ${newlang}.normproto
mv pffmtable ${newlang}.pffmtable
mv shapetable ${newlang}.shapetable
mv inttemp ${newlang}.inttemp
mv unicharset ${newlang}.unicharset
combine_tessdata ${newlang}.
cp ${newlang}.traineddata /usr/local/share/tessdata/
{% endhighlight %}

To recognize characters in a new image simply type

    tesseract image.tif output -l newfra

[Full Tesseract specification](https://code.google.com/p/tesseract-ocr/wiki/TrainingTesseract3)
