
---
layout: post
title:  "Training optical caracter recognition technology Tesseract on a new character font on MacOS"
date:   2015-07-16 23:00:51
categories: character recognition
---

#Training Tesseract on a new font

First install Tesseract

    brew install tesseract

Tesseract will use a **TIFF image file** (with characters to learn) and a **Box file** (indicating the bounding box of the characters in the image) to do its training to a new language.

Here is an example of TIFF file :

![TIFF File](https://printalert.files.wordpress.com/2014/04/ocr_input.jpg)

The format of the box file is one ligne per character in the image and each line of the form `char bl_x bl_y rt_x rt_y` where `char` is the character, `bl_x` the abcyss of bottom-left corner in a coordinate system where (0,0) is at the bottom-left corner of the TIFF image.

To create the box file, it's possible to use Tesseract recognition engine and add manually complete the lines that were not recognized automatically. To perform a better recognition, you can download additional languages such as "fra" for French language. Put the file `fra.traineddata` in `/usr/local/share/tessdata/` for Tesseract to use it.

First begin by creating the character table as a TIFF image. Then, here is the list of command to create a new language for Tesseract from this TIFF image :

{% highlight bash %}
newlang="newfra"
tesseract ${newlang}.std.exp0.tif ${newlang}.std.exp0 batch.nochop makebox # the option '-l fra' to use the French language
tesseract ${newlang}.std.exp0.tif ${newlang}.std.exp0 box.train.stderr
unicharset_extractor ${newlang}.std.exp0.box
shapeclustering -F font_properties -U unicharset ${newlang}.std.exp0.tr
mftraining -F font_properties -U unicharset -O lfra.unicharset ${newlang}.std.exp0.tr
cntraining ${newlang}.std.exp0.tr
mv normproto ${newlang}.normproto
mv pffmtable ${newlang}.pffmtable
mv shapetable ${newlang}.shapetable
mv inttemp ${newlang}.inttemp
mv unicharset ${newlang}.unicharset
combine_tessdata ${newlang}.
{% endhighlight %}

Put the file `newfra.traineddata` in `/usr/local/share/tessdata/` for Tesseract to use it. To recognize characters in a new image simply type

    tesseract image.tif output -l newfra
