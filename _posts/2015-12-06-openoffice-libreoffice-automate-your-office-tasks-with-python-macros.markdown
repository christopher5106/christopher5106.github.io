---
layout: post
title:  "Interface-oriented in OpenOffice / LibreOffice : automate your office tasks with Python Macros"
date:   2015-12-06 23:00:51
categories: office
---

# OpenOffice or LibreOffice ?

OpenOffice and LibreOffice are the main open-source office suites.

LibreOffice was a fork of OpenOffice.org (when OpenOffice went under Oracle's umbrella) and is built on the original OpenOffice.org code base.

Both are equivalent, but the usual advise is to use LibreOffice ([see the differences](http://www.howtogeek.com/187663/openoffice-vs.-libreoffice-whats-the-difference-and-which-should-you-use/)) since it is the project of the volunteers of the open-source community and has been developping more quickly.

I'll speak about LibreOffice now, but the same is true for OpenOffice.

# Which language choice for writing your LibreOffice macros ?

Macros are scripting for the office suite.

Many languages are accepted by the [LibreOffice API](http://api.libreoffice.org/), thanks to the [Universal Network Objects (UNO)](https://en.wikipedia.org/wiki/Universal_Network_Objects). Among the available languages : Visual Basic, Java, C/C++, Javascript, Python.

The API is interface-oriented, meaning you communicate with the controller of the interface and the document has to be open. Many other Python libraries are not interface-oriented, creating  directly the file in the **Open Document format** and saving it to disk with the correct extension

- **.odt** for text files
- **.ods** for spreadsheets
- **.odp** for presentations
- **.odg** for drawings


For the choice of the language, I would first insist on the **multi-platform requirement**, which means it's better if the macro / script can be executed on different platforms such as Windows, Mac OS or Linux, because LibreOffice is also multi-platform and documents will be shared between users from which we cannot expect a particular platform. Visual Basic is not multi-platform and would require significant changes from one plateform to another (Visual Basic, [Real Basic](http://www.xojo.com/), AppleScript...).

Java and C/C++ require compilation, are much more complex and verbose.

For a scripting need, I would advise Javascript or Python. Both are very present in script development world wide and are standard de facto. Many tools have been built for task automation on Javascript, such as Cordova (the multi-platform mobile app framework) or Grunt. Many other tools are using Python as well, such as AWS CLI for example.

But, Javascript could be **not precise enough** (even though there exists very nice libraries for [numeric computation](http://blog.smartbear.com/testing/four-serious-math-libraries-for-javascript/)) and could be disconcerting for your Office users due to rounding errors ( `0.1 + 0.2` does not equals `0.3` in Javascript).

On the contrary, **Python has been used extensively for numeric computation**, with famous libraries such as Numpy, Numexpr ... which make it perfect for spreadsheet macros.

Python has also numerous available libraries for other purposes, due to its success and support from big digital companies, such as Excel reading or writing libraries which make it the perfect choice for macro development.

Even though Python 2.7 still remains very used, and Python 3 introduced differences, the latest version of LibreOffice comes with Python 3.3, so the use of Python 3.3 is advised for durability.

{% highlight bash %}
/Applications/LibreOffice.app/Contents/MacOS/python --version
#Python 3.3.5
{% endhighlight %}

# First play with the shell before creating your own macro

Before creating your own macro for , you can play with the Python shell.

First launch LibreOffice Calc (to create spreadsheet open document) with an open socket to communicate with from the shell, on your Mac OS :

    /Applications/LibreOffice.app/Contents/MacOS/soffice --calc --accept="socket,host=localhost,port=2002;urp;StarOffice.ServiceManager"

(for the Windows command : `c:\Program Files\OpenOffice1.1\program\soffice "--calc --accept=socket,host=localhost,port=2002;urp;"` but if any trouble, have a look the [proposed workarounds](http://www.openoffice.org/udk/python/python-bridge.html)).

and launch the Python shell

    /Applications/LibreOffice.app/Contents/MacOS/python

To initialize your context, type the following lines in your python shell :

{% highlight python %}
import socket  # only needed on win32-OOo3.0.0
import uno

# get the uno component context from the PyUNO runtime
localContext = uno.getComponentContext()

# create the UnoUrlResolver
resolver = localContext.ServiceManager.createInstanceWithContext(
				"com.sun.star.bridge.UnoUrlResolver", localContext )

# connect to the running office
ctx = resolver.resolve( "uno:socket,host=localhost,port=2002;urp;StarOffice.ComponentContext" )
smgr = ctx.ServiceManager

# get the central desktop object
desktop = smgr.createInstanceWithContext( "com.sun.star.frame.Desktop",ctx)

# access the current writer document
model = desktop.getCurrentComponent()

{% endhighlight %}

These lines are common for every documents. Now you can interact with the document. Since we launched LibreOffice with `--calc` option, it's a spreadsheet.


{% highlight python %}
# access the active sheet
active_sheet = model.CurrentController.ActiveSheet

# access cell C4
cell1 = active_sheet.getCellRangeByName("C4")

# set text inside
cell1.String = "Hello world"

# other example with a value
cell2 = oSheet.getCellRangeByName("E6")
cell2.Value = cell2.Value + 1
{% endhighlight %}

If it's a text document, you can try the following interactions :

{% highlight python %}
# access the document's text property
text = model.Text

# create a cursor
cursor = text.createTextCursor()

# insert the text into the document
text.insertString( cursor, "Hello World", 0 )
{% endhighlight %}

Here is a schema for what we've just done : the shell communicates with the LibreOffice runtime to command actions inside the current document.

![Python Uno mode ipc](http://www.openoffice.org/udk/python/images/mode_ipc.png)

# Create your first macro

It is the other mode, the macro is called from inside the Libreoffice program :

![Python Uno mode component](http://www.openoffice.org/udk/python/images/mode_component.png)

OpenOffice.org does not offer a way to edit Python scripts. You have to use your own text editor (such as Sublim, Atom...) and your own commands.

There are 3 places where you can put your code. The first way is to add it as a library for LibreOffice in one of the directories in the PYTHONPATH

{% highlight python %}
import sys
for i in sys.path:
     print(i)
{% endhighlight %}

which gives

    /Applications/LibreOffice.app/Contents/Resources
    /Applications/LibreOffice.app/Contents/Frameworks
    /Applications/LibreOffice.app/Contents/Frameworks/LibreOfficePython.framework/Versions/3.3/lib/python3.3
    /Applications/LibreOffice.app/Contents/Frameworks/LibreOfficePython.framework/Versions/3.3/lib/python3.3/lib-dynload
    /Applications/LibreOffice.app/Contents/Frameworks/LibreOfficePython.framework/Versions/3.3/lib/python3.3/lib-tk
    /Applications/LibreOffice.app/Contents/Frameworks/LibreOfficePython.framework/Versions/3.3/lib/python3.3/site-packages
    /Applications/LibreOffice.app/Contents/Frameworks/LibreOfficePython.framework/lib/python33.zip
    /Applications/LibreOffice.app/Contents/Frameworks/LibreOfficePython.framework/lib/python3.3
    /Applications/LibreOffice.app/Contents/Frameworks/LibreOfficePython.framework/lib/python3.3/plat-darwin
    /Applications/LibreOffice.app/Contents/Frameworks/LibreOfficePython.framework/lib/python3.3/lib-dynload


But this is only useful to be used in other macros.

The 2 other ways are to insert your script

- either globally on your computer, in your local LibreOffice installation,

- or inside the document, so that when shared another computer (by email, or whatever means), the document has still functional macros.

Let's see how to install it in the LibreOffice install first, I'll show you the document-inside install in the next section.

You can find and call your Macro scripts from the LibreOffice menu for macros **Tools > Macros > Organize Macros**.

![LibreOffice Python Macro Directory]({{ site.url }}/img/libreoffice_python_macro_directory.png)

If you get a "Java SE 6 Error message" such as bellow

![JavaSE6]({{ site.url }}/img/JavaSE6.png)

download the [Java SE 6 version here](http://download.info.apple.com/Mac_OS_X/031-03190.20140529.Pp3r4/JavaForOSX2014-001.dmg).


Let's edit a first macro script file **myscript.py**, we need to create a method *HelloWorldPython* :

{% highlight python %}
def HelloWorldPython( ):
    """Prints the string 'Hello World(in Python)' into the current document"""
#get the doc from the scripting context which is made available to all scripts
    desktop = XSCRIPTCONTEXT.getDesktop()
    model = desktop.getCurrentComponent()
#check whether there's already an opened document. Otherwise, create a new one
    if not hasattr(model, "Text"):
        model = desktop.loadComponentFromURL(
            "private:factory/swriter","_blank", 0, () )
#get the XText interface
    text = model.Text
#create an XTextRange at the end of the document
    tRange = text.End
#and set the string
    tRange.String = "Hello World (in Python)"
    return None
{% endhighlight %}

and copy it to the Macro directory for LibreOffice :

	cp myscript.py /Applications/LibreOffice.app/Contents/Resources/Scripts/Python/

which you can run from :

![LibreOffice Python Macro Directory]({{ site.url }}/img/libreoffice_python_macro_script.png)

after having opened a text document.

In case there are multiple methods, all of them will be exported, but we can also specify which one to export with the following statement at the end of the file :

{% highlight python %}
g_exportedScripts = capitalisePython,
{% endhighlight %}

For distribution of code, [OXT format](http://wiki.openoffice.org/wiki/Documentation/DevGuide/Extensions/Extensions) acts as containers of code that will be installed by the Extension Manager or with the command line `/Applications/LibreOffice.app/Contents/MacOS/unopkg`.







# Pack your script inside the document : the OpenDocument format

OpenDocument files are zipped directories.

You can have a look at inside by creating and saving a opendocument spreadsheet document with LibreOffice and then unzipping it :

    unzip Documents/test.ods -d test

You'll get the following list of files and subdirectories in your extracted file :

    ├── Configurations2
    │   ├── accelerator
    │   │   └── current.xml
    │   ├── floater
    │   ├── images
    │   │   └── Bitmaps
    │   ├── menubar
    │   ├── popupmenu
    │   ├── progressbar
    │   ├── statusbar
    │   ├── toolbar
    │   └── toolpanel
    ├── META-INF
    │   └── manifest.xml
    ├── Thumbnails
    │   └── thumbnail.png
    ├── content.xml
    ├── manifest.rdf
    ├── meta.xml
    ├── mimetype
    ├── settings.xml
    └── styles.xml

You can directly append your script to the file with the *zipfile library* :

{% highlight python %}
import zipfile
doc = zipfile.ZipFile("Documents/test.ods", 'a')
doc.write("myscript.py", "Scripts/python/myscript.py")
doc.close()

manifest = []
for line in doc.open('META-INF/manifest.xml'):
    if '</manifest:manifest>' in line:
        for path in ['Scripts/','Scripts/python/','Scripts/python/myscript.py']:
            manifest.append(' <manifest:file-entry manifest:media-type="application/binary" manifest:full-path="%s"/>' % path)
    manifest.append(line)
doc.writestr('META-INF/manifest.xml', ''.join(manifest))
{% endhighlight %}

**Well done !**
