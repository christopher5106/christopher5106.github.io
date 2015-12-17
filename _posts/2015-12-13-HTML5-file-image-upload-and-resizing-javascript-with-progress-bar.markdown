---
layout: post
title:  "HTML5 and Javascript: file upload with progress bar, client-side image resizing and multiple runtimes"
date:   2015-12-13 23:00:51
categories: web
---

There are tons of libraries on the web, such as [shown in this list](http://designscrazed.org/html5-jquery-file-upload-scripts/), but these libraries are always much more complicated than needed, and modifying them will require 10 times more work than do it from scratch.

So let us see the different components to do our own file uploader script.

# Which request ?

For files, it's necessarily a POST request (passing the file in the parameters of a GET request would be possible for small files though but a very bad anti-pattern).

There exists different encoding format for the content of the data :

- application/x-www-form-urlencoded
- text/plain
- multipart/form-data

The `multipart/form-data` type is the recommended one for files since you can upload multiple files and chunks of files.

In the form tag element `<form>`, the format is usually specified by the `enctype` attribute and the correct request is made by the browser on input change. Default is `application/x-www-form-urlencoded`.


# The XMLHttpRequest Object and the progress status

XMLHttpRequest enables to send a HTTP Request to server in Javascript and is used heavily in AJAX programming.

{% highlight html %}
<script type="text/javascript">
var xhr = new XMLHttpRequest();
xhr.open("GET", "http://christopher5106.github.io/img/mac_digits.png?" + new Date().getTime());
xhr.onprogress = function (e) {
    if (e.lengthComputable) {
        console.log(e.loaded+  " / " + e.total)
    }
}
xhr.onloadstart = function (e) {
    console.log("start")
}
xhr.onloadend = function (e) {
    console.log("end")
}
xhr.send();
</script>
{% endhighlight %}


will produce the following output in the console

    start
    462 / 3509517
    9894 / 3509517
    30854 / 3509517
    70678 / 3509517
    150326 / 3509517
    281326 / 3509517
    437478 / 3509517
    593630 / 3509517
    655462 / 3509517
    825238 / 3509517
    1021214 / 3509517
    1227670 / 3509517
    1447750 / 3509517
    1677262 / 3509517
    1876382 / 3509517
    2080742 / 3509517
    2291390 / 3509517
    2477934 / 3509517
    2706398 / 3509517
    2977830 / 3509517
    3232494 / 3509517
    3485062 / 3509517
    3509517 / 3509517
    end

Another way to write it for **GET requests** is using `xhr.addEventListener("progress", updateProgress); xhr.addEventListener("load", transferComplete); xhr.addEventListener("error", transferFailed); xhr.addEventListener("abort", transferCanceled)`.

For **POST requests**, you need to monitor also the upload progress with :

{% highlight javascript %}
xhr.upload.addEventListener("progress", function(evt){
      if (evt.lengthComputable) {
        console.log("add upload event-listener" + evt.loaded + "/" + evt.total);
      }
    }, false);
{% endhighlight %}

I would advise the use of a HTML `<progress>` element to display current progress.

Using jQuery

{% highlight javascript %}
$.ajax({
  xhr: function()
  {
    var xhr = new window.XMLHttpRequest();
    //Upload progress
    xhr.upload.addEventListener("progress", function(evt){
      if (evt.lengthComputable) {
        var percentComplete = evt.loaded / evt.total;
        //Do something with upload progress
        console.log(percentComplete);
      }
    }, false);
    //Download progress
    xhr.addEventListener("progress", function(evt){
      if (evt.lengthComputable) {
        var percentComplete = evt.loaded / evt.total;
        //Do something with download progress
        console.log(percentComplete);
      }
    }, false);
    return xhr;
  },
  type: 'POST',
  url: "/",
  data: {},
  success: function(data){
    //Do something success-ish
  }
});
{% endhighlight %}

# The FormData Element

The FormData element simplifies the creation of a POST request of type multipart/form-data ([demonstration here](https://developer.mozilla.org/en-US/docs/Web/API/XMLHttpRequest/Using_XMLHttpRequest#Submitting_forms_and_uploading_files)) and the call to send a form is simply :

{% highlight javascript %}
var xhr = new XMLHttpRequest();
xhr.open("post", formElement.action);
xhr.send(new FormData(formElement));
{% endhighlight %}

It uses the XMLHttpRequest method send() to send the form's data.

The FormData is recommended not only for forms, but for any key-value post purpose, by [creating a FormData object from scratch](https://developer.mozilla.org/en-US/docs/Web/API/FormData/Using_FormData_Objects) :

{% highlight javascript %}
var formData = new FormData();
formData.append("username", "Groucho");
formData.append("accountnum", 123456);
var request = new XMLHttpRequest();
request.open("POST", "http://foo.com/submitform.php");
request.send(formData);
{% endhighlight %}

Using FormData with jQuery :

{% highlight html %}
<script src="https://ajax.googleapis.com/ajax/libs/jquery/2.1.4/jquery.min.js"></script>
<script>
$.ajax({
     url: event.url,
     data: data,
     cache: false,
     contentType: false,
     processData: false,
     type: 'POST',
     success: function(data){
        ... handle errors...
     }
 });
 </script>
{% endhighlight %}


# The Blob and File objects

Let's use [PutsReq service](http://putsreq.com/) that is a free file bin to debug our POST requests and send it a file :

{% highlight html %}
<input name="imagefile[]" type="file" id="takePictureField" accept="image/*" onchange="uploadPhotos('http://putsreq.com/jX2tGa272jPmLH4KtR2n')" />
<script type="text/javascript">
window.uploadPhotos = function(url){
  var formData = new FormData();

  // HTML file input, chosen by user
  var fileInputElement = document.getElementById("takePictureField");
  formData.append("userfile", fileInputElement.files[0]);

  // JavaScript file-like object
  var content = '<a id="a"><b id="b">hey!</b></a>'; // the body of the new file...
  var blob = new Blob([content], { type: "text/xml"});
  formData.append("webmasterfile", blob);

  var xhr = new XMLHttpRequest();
  xhr.open("POST", url);
  xhr.send(formData);
}
</script>
{% endhighlight %}


The field "webmasterfile" is a Blob. A Blob object represents a file-like object of immutable, raw data.

The File interface is based on Blob, inheriting blob functionality and expanding it to support files on the user's system.


# Resize image size client-side with FileReader API


Taken from [here](http://stackoverflow.com/questions/23945494/use-html5-to-resize-an-image-before-upload), here is a full upload with a resize :

{% highlight html %}
<input name="imagefile[]" type="file" id="takePictureField" accept="image/*" onchange="uploadPhotos('http://putsreq.com/jX2tGa272jPmLH4KtR2n')" />

<script src="https://ajax.googleapis.com/ajax/libs/jquery/2.1.4/jquery.min.js"></script>
<script type="text/javascript">
window.uploadPhotos = function(url){
  console.log("Upload to URL " + url)
    // Read in file
    var file = event.target.files[0];

    // Ensure it's an image
    if(file.type.match(/image.*/)) {
        console.log('An image has been loaded');

        // Load the image
        var reader = new FileReader();
        reader.onload = function (readerEvent) {
            var image = new Image();
            image.onload = function (imageEvent) {

                // Resize the image
                var canvas = document.createElement('canvas'),
                    max_size = 544,
                    width = image.width,
                    height = image.height;
                if (width > height) {
                    if (width > max_size) {
                        height *= max_size / width;
                        width = max_size;
                    }
                } else {
                    if (height > max_size) {
                        width *= max_size / height;
                        height = max_size;
                    }
                }
                canvas.width = width;
                canvas.height = height;
                canvas.getContext('2d').drawImage(image, 0, 0, width, height);
                var dataUrl = canvas.toDataURL('image/jpeg');
                var resizedImage = dataURLToBlob(dataUrl);
                $.event.trigger({
                    type: "imageResized",
                    blob: resizedImage,
                    url: url
                });
            }
            image.src = readerEvent.target.result;
        }
        reader.readAsDataURL(file);
    }
};

/* Utility function to convert a canvas to a BLOB */
var dataURLToBlob = function(dataURL) {
  console.log("DataURLToBlob")
    var BASE64_MARKER = ';base64,';
    if (dataURL.indexOf(BASE64_MARKER) == -1) {
        var parts = dataURL.split(',');
        var contentType = parts[0].split(':')[1];
        var raw = parts[1];

        return new Blob([raw], {type: contentType});
    }

    var parts = dataURL.split(BASE64_MARKER);
    var contentType = parts[0].split(':')[1];
    var raw = window.atob(parts[1]);
    var rawLength = raw.length;

    var uInt8Array = new Uint8Array(rawLength);

    for (var i = 0; i < rawLength; ++i) {
        uInt8Array[i] = raw.charCodeAt(i);
    }

    return new Blob([uInt8Array], {type: contentType});
}
/* End Utility function to convert a canvas to a BLOB      */

/* Handle image resized events */
$(document).on("imageResized", function (event) {
  console.log("imageResized")
    var data = new FormData();
    if (event.blob && event.url) {

        data.append('file', event.blob);
        $.ajax({
            url: event.url,
            data: data,
            cache: false,
            contentType: false,
            processData: false,
            type: 'POST',
            success: function(data){
               console.log("Uploaded")
            }
        });
    }
});
</script>

{% endhighlight %}


You can also add drag-and-drop functionality very easily following this [tutorial](http://html5doctor.com/drag-and-drop-to-server/).

Lastly, you can also, during a drag-and-drop of an image from another browser window, get the URL of the image to send to the server : 

	var url = event.dataTransfer.getData('URL');


