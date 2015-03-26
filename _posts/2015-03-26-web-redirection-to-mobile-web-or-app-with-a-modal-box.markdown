---
layout: post
title:  "Redirection to the mobile app or the mobile website with a modal box"
date:   2015-03-26 23:00:51
categories: mobile
---

The idea is to open a modal box when the website (WWW) is viewed on the mobile phone or a tablet, to propose the user a redirection to the mobile app or the mobile website (M), which have a better experience on small devices.

I won't explain in this article the advantages of keeping two separate WWW and M web site. I precise the technical constraints and possibilities for the different platforms, in particular Android and IOS. 

This article is true when **the mobile app and the mobile website have the same ergonomy**, for example in the case of an hybrid app.

Here is my best practice.

Such a redirection can be done by a simple and standalone script in Javascript. 

First, detecting of the mobile device and setting the variables : 

{% highlight javascript %}
var IS_IPAD = navigator.userAgent.match(/iPad/i) != null,
IS_IPHONE = !IS_IPAD && ((navigator.userAgent.match(/iPhone/i) != null) || (navigator.userAgent.match(/iPod/i) != null)),
IS_IOS = IS_IPAD || IS_IPHONE,
IS_ANDROID = !IS_IOS && navigator.userAgent.match(/android/i) != null,
IS_MOBILE = IS_IOS || IS_ANDROID;
}
{% endhighlight %}

#Mapping URL between WWW and M web sites

No need to say that the main advantage of keeping two separate web sites, one for PC, and one for mobile, is to offer different but better experiences, and that will usually lead to different URL.

In a perfect world, one could think only the domain name changes, and the rule could simply be that `http://www.domain.com + path` is equivalent to `http://m.domain.com + path`. Nevertheless it's quite utopic. And also more often, there are some pages for which there is no equivalent on mobile website and vice-versa.

To make the mapping between the WWW and M web sites, I prefer to advise that in each HTML page or template of the WWW website, where the script will be inserted, a `path` parameter has to be inserted in the page for the redirection to be proposed to the user, or not, and where to redirect. This path parameter will enable to construct either : 

* the mobile website URL : `http://m.my-domain.com/ + path`

* the mobile app URI scheme (usually called "custom URI scheme") : `my-app:// + path ` that can be used either to directly launch the app if the app is installed, or for the [smart banner](https://developer.apple.com/library/mac/documentation/AppleApplications/Reference/SafariWebContent/PromotingAppswithAppBanners/PromotingAppswithAppBanners.html).

But do we really need to create a parameter ? 

No, not really. We can combine it with the Applink tag in the HTML page, and detect it :


{% highlight javascript %}
if( (typeof $("meta[property='al:android:url']").attr("content") != "undefined" || typeof $("meta[property='al:ios:url']").attr("content") != "undefined") && IS_MOBILE) {
  if(getCookie("appli")!="") open(true);
  else $('#ask_for_app').modal('show');
}
{% endhighlight %}

**In conclusion, you just need to insert the JS script in the HTML header of all pages, and add the Applink tag where a mobile redirection can be done. The presence of the Applink will decide if there is a redirection.**


#Applinks

[Applinks](http://applinks.org/) allow a page, when shared on a social network, and seen in the feed of the social networks, to directly open the app if the app is installed, instead of the browser.

**It is not possible for a website to check if an app has been installed. But it's possible for a native app to check if the app is installed.**

The other good thing about Applinks is that it is an open-source and cross-platform standard, that can be implemented in any app that deals with URL and webpages. [Applinks are in particular implemented by Facebook](https://developers.facebook.com/docs/applinks). There also exists a [Cordova plugin](https://github.com/francimedia/phonegap-applinks).

Here is an example of 

{% highlight html %}
<meta property="al:ios:url" content="my-app://path" />
<meta property="al:android:url" content="my-app://path">
<meta property="al:ios:app_store_id" content="apple-app-id" />
<meta property="al:android:package" content="google-app-package">
<meta property="al:android:app_name" content="My App">
<meta property="al:ios:app_name" content="My App" />
<meta property="og:title" content="example page title" />
<meta property="og:type" content="website" />
{% endhighlight %}


#Custom URI schemes

The custom URI schemes are URI with a "custom" protocol, for example `my-app://my-page`.

This protocol becomes active on the mobile phone when the user has installed the corresponding app.

It enables to target a precise page in the app. 

Custom URI schemes re-create a sort of "hyperlinks" for mobile apps, as on the web with hypertext links. 




#The script

So, let's assume the script has detected that 

- the Applink tag is present

- the user is viewing the page on a mobile phone

there is a potential redirection to propose to the user by opening the modal box to the user.

What are the "call to action" buttons to propose the user ?

It is where it becomes tricky because : 

- on Android phones, it is possible to redirect the user to the mobile app, and if the mobile app is not installed, the user will be automatically redirected to Google Play. This can be done with a simple INTENT action: 

{% highlight javascript %}
		window.location = 'intent:/'+$("meta[property='al:android:url']").attr("content").split(':/')[1]+'#Intent;package='+$("meta[property='al:android:package']").attr("content")+';scheme='+$("meta[property='al:android:url']").attr("content").split(':/')[0]+';launchFlags=268435456;end;';
{% endhighlight %}

that can be proposed under a "Download the app" button. 

- on iPhone, it's possible to do the same with : 

{% highlight javascript %}
window.location = $("meta[property='al:ios:url']").attr("content");

setTimeout(function() {

	// If the user is still here, open the App Store
	if (!document.webkitHidden) {

    window.location = 'http://itunes.apple.com/app/id' + $("meta[property='al:ios:app_store_id']").attr("content") ;
    }
}, 25);
{% endhighlight %}

If the app is already installed (with its custom URI shemes), it's going to launch the app at the correct page. But if the app is not installed, the user will very shortly see an error popup, and be redirected to the AppStore with the `setTimeout function`. Not very good, this popup, but we have no other choice. 


#Download or open the app ?

The "Download the app" button is not very nice when the user has already installed the app ! In particular because this user might come many times to the WWW website on his phone. The same problem occurs on both IOS and Android devices. It is due to the fact it is not possible to know from a web page if an app has been installed, for security reasons.

To avoid this situation, the mainstream solution is to ask a question at the first time and then set a cookie, not to ask it again.

The way to ask it can vary from one site to another : 

- "Already installed ?"

- "Open the app" / "Download the app"

- ...

Here is my full script : 

{% highlight javascript %}

var MOBILE_BASE_URL = "http://m.my-domain.com"

function getCookie(cname) {
  var name = cname + "=";
  var ca = document.cookie.split(';');
  for(var i=0; i<ca.length; i++) {
    var c = ca[i];
    while (c.charAt(0)==' ') c = c.substring(1);
    if (c.indexOf(name) == 0) return c.substring(name.length,c.length);
  }
  return "";
}

function setCookie(cname, cvalue, exdays) {
  var d = new Date();
  d.setTime(d.getTime() + (exdays*24*60*60*1000));
  var expires = "expires="+d.toUTCString();
  document.cookie = cname + "=" + cvalue + "; " + expires;
}

var IS_IPAD = navigator.userAgent.match(/iPad/i) != null,
IS_IPHONE = !IS_IPAD && ((navigator.userAgent.match(/iPhone/i) != null) || (navigator.userAgent.match(/iPod/i) != null)),
IS_IOS = IS_IPAD || IS_IPHONE,
IS_ANDROID = !IS_IOS && navigator.userAgent.match(/android/i) != null,
IS_MOBILE = IS_IOS || IS_ANDROID;

function open(has_appli) {
  // If it's not an universal app, use IS_IPAD or IS_IPHONE
  setCookie("appli",true,1000);

  if (IS_IOS) {
    if(has_appli) {
      window.location = $("meta[property='al:ios:url']").attr("content") || $("meta[property='al:android:url']").attr("content");

      setTimeout(function() {
        // If the user is still here, open the App Store
        if (!document.webkitHidden) {
          window.location = 'http://itunes.apple.com/app/id' + $("meta[property='al:ios:app_store_id']").attr("content");
        }
      }, 25);
    } else { 
    	window.location = 'http://itunes.apple.com/app/id'+$("meta[property='al:ios:app_store_id']").attr("content");
    }

  } else if (IS_ANDROID) {
    window.location = 'intent:/'+$("meta[property='al:android:url']").attr("content").split(':/')[1]+'#Intent;package='+$("meta[property='al:android:package']").attr("content")+';scheme='+$("meta[property='al:android:url']").attr("content").split(':/')[0]+';launchFlags=268435456;end;';
  }
}

$("#yes").click(function() { open(true); } );
$("#no").click(function() { open(false); } );
$("#mobile").click(function() { 
	window.location = MOBILE_BASE_URL + "/" + $("meta[property='al:android:url']").attr("content").split(':/')[1]; 
} );

if( (typeof $("meta[property='al:android:url']").attr("content") != "undefined" || typeof $("meta[property='al:ios:url']").attr("content") != "undefined") && IS_MOBILE) {
  if(getCookie("appli")!="") 
  	open(true);
  else 
  	$('#ask_for_app').modal('show');
}

{% endhighlight %}