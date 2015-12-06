---
layout: post
title:  "OpenOffice / LibreOffice : automate your office task with Python Macros"
date:   2015-12-06 23:00:51
categories: office
---

# OpenOffice or LibreOffice ?

OpenOffice and LibreOffice are the main open-source office suites.

LibreOffice was a fork of OpenOffice.org (when OpenOffice went under Oracle's umbrella) and is built on the original OpenOffice.org code base.

Both are equivalent, but the usual advise is to use LibreOffice ([see the differences](http://www.howtogeek.com/187663/openoffice-vs.-libreoffice-whats-the-difference-and-which-should-you-use/)) since it is the project of the volunteers of the open-source community and has been developping more quickly.

I'll speak about LibreOffice now, but the same is true for OpenOffice.

# Macros

Macros are scripting for the office suite.

Many languages are accepted by [LibreOffice API](http://api.libreoffice.org/), thanks to the [Universal Network Objects (UNO)](https://en.wikipedia.org/wiki/Universal_Network_Objects). Among the available languages : Visual Basic, Java, C/C++, Javascript, Python.

It is interface-oriented, meaning you communicate with the controller of the interface and the document has to be open. Many other Python libraries are not interface-oriented, creating the ODS file directly and saving it to disk.

For the choice of the language, I would first insist on the multi-platform requirement, which means it's better if the macro / script can be executed on different platforms such as Windows, Mac OS or Linux. Visual Basic is not multi-platform and would require significant changes from one plateform to another (Visual Basic, [Real Basic](http://www.xojo.com/), AppleScript...).

Java and C/C++ require compilation, are much more complex and verbose.

For a scripting need, I would advise Javascript or Python. Both are very present in script development world wide, they are standard de facto. Many tools have been built for task automation on Javascript, such as Cordova (the multi-platform mobile app framework) or Grunt. Many other tools are using Python as well, such as AWS CLI for example.

But, Javascript could not be precise enough (even though there exists very nice libraries for [numeric computation](http://blog.smartbear.com/testing/four-serious-math-libraries-for-javascript/)) and could be disconcerting for your Office users due to rounding errors ( `0.1 + 0.2` does not equals `0.3` in Javascript).

On the contrary, Python has been used extensively for numeric computation, with famous libraries such as Numpy, Numexpr ... Python is also the perfect choice because of its numerous libraries, such as Excel reading or writing libraries.

Even though Python 2.7 still remains very used, and Python 3 introduced differences, LibreOffice comes with Python 3.3, so Python 3.3 is advised for durability.
