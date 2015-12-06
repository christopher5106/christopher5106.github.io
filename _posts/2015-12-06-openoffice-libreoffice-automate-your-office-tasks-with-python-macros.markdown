---
layout: post
title:  "OpenOffice / LibreOffice : automate your office tasks with Python Macros"
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

Many languages are accepted by the [LibreOffice API](http://api.libreoffice.org/), thanks to the [Universal Network Objects (UNO)](https://en.wikipedia.org/wiki/Universal_Network_Objects). Among the available languages : Visual Basic, Java, C/C++, Javascript, Python.

The API is interface-oriented, meaning you communicate with the controller of the interface and the document has to be open. Many other Python libraries are not interface-oriented, creating  directly the file in the **Open Document format** and saving it to disk with the correct extension

- **.odt** for text files
- **.ods** for spreadsheets
- **.odp** for presentations
- **.odg** for drawings


For the choice of the language, I would first insist on the **multi-platform requirement**, which means it's better if the macro / script can be executed on different platforms such as Windows, Mac OS or Linux, because LibreOffice is also multi-platform. Visual Basic is not multi-platform and would require significant changes from one plateform to another (Visual Basic, [Real Basic](http://www.xojo.com/), AppleScript...).

Java and C/C++ require compilation, are much more complex and verbose.

For a scripting need, I would advise Javascript or Python. Both are very present in script development world wide and are standard de facto. Many tools have been built for task automation on Javascript, such as Cordova (the multi-platform mobile app framework) or Grunt. Many other tools are using Python as well, such as AWS CLI for example.

But, Javascript could be **not precise enough** (even though there exists very nice libraries for [numeric computation](http://blog.smartbear.com/testing/four-serious-math-libraries-for-javascript/)) and could be disconcerting for your Office users due to rounding errors ( `0.1 + 0.2` does not equals `0.3` in Javascript).

On the contrary, **Python has been used extensively for numeric computation**, with famous libraries such as Numpy, Numexpr ... which make it perfect for spreadsheet macros.

Python has numerous available libraries, such as Excel reading or writing libraries which make it the perfect choice for macro development.

Even though Python 2.7 still remains very used, and Python 3 introduced differences, the latest version of LibreOffice comes with Python 3.3, so the use of Python 3.3 is advised for durability.

     /Applications/LibreOffice.app/Contents/MacOS/python --version
     #Python 3.3.5

# First play with the shell

Before creating your own macro for a LibreOffice spreadsheet, you can play with the Python shell.

First launch LibreOffice with the socket, here is the Mac OS command :

    /Applications/LibreOffice.app//Contents/MacOS/soffice --calc --accept="socket,host=localhost,port=2002;urp;StarOffice.ServiceManager"

(for the Windows command : `c:\Program Files\OpenOffice1.1\program\soffice "--calc --accept=socket,host=localhost,port=2002;urp;"`)

and launch Python

    /Applications/LibreOffice.app/Contents/MacOS/python

To initialize your context, type the following lines in your python shell :

{% hightlight python %}
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


{% hightlight python %}
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

{% hightlight python %}
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

Let's edit a first macro script file **mymacro.py** :




There are 3 places where you can put your code :

- in a library for LibreOffice in one of the directories in the PYTHONPATH

{% hightlight python %}
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

- in the Macro directory for LibreOffice, so that you can call your Macro script from the LibreOffice menu : ** Tools > Macros > Organize Macros **

    /Applications/LibreOffice.app/Contents/Resources/Scripts/Python

- inside the document, so that when shared another computer (by email, or whatever means), the document has still functional macros. I'll show you this in the next section.


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

**Well done !**
