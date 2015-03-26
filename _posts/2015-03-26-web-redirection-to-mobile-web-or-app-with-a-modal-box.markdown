---
layout: post
title:  "Redirection to mobile website or mobile app with a modal box"
date:   2015-03-26 23:00:51
categories: mobile
---

The idea is to show a modal box when the website www is called on the mobile phone or a tablet, to propose the mobile app or the mobile website (M) to the user, and offer her a better experience. 

I won't explain in this article the advantages of keeping two separate www and M web site. I precise the constraints and possibilities of the different platforms, in particular Android and IOS.

This can be done by a simple and standalone script in Javascript. First, the detection of the mobile device : 

{% highlight javascript %}
var IS_IPAD = navigator.userAgent.match(/iPad/i) != null,
IS_IPHONE = !IS_IPAD && ((navigator.userAgent.match(/iPhone/i) != null) || (navigator.userAgent.match(/iPod/i) != null)),
IS_IOS = IS_IPAD || IS_IPHONE,
IS_ANDROID = !IS_IOS && navigator.userAgent.match(/android/i) != null,
IS_MOBILE = IS_IOS || IS_ANDROID;
}
{% endhighlight %}

#Mapping between M web site and www web site

No need to say that the main advantage of having two web sites, one for PC, and one for mobile, is to have different experiences and though different URL.

In a perfect world, one could think only the domain name changes, and the rule could simply be that `http://www.domain.com +path` is equivalent to `http://m.domain.com + path`. Nevertheless it's quite utopic.

To make the mapping between M web site and www web site, I prefer to advise that in each HTML page or template of the www website, where the script will be inserted, a `path` parameter has to be inserted in the page for the redirection to proposed to the user, otherwise the redirection won't be proposed. This path parameter will enable to construct either : 

* the mobile website URL : `http://m.my-domain.com/ + path`

* the mobile app URI scheme (usually called "custom URI scheme") : `my-app:// + path ` that can be used either to directly launch the app if the app is installed, or for the [smart banner](https://developer.apple.com/library/mac/documentation/AppleApplications/Reference/SafariWebContent/PromotingAppswithAppBanners/PromotingAppswithAppBanners.html) 

But do we really need to create a parameter ? 

No, not really. We can combine it with the applink parameter. 


{% highlight javascript %}
if( (typeof $("meta[property='al:android:url']").attr("content") != "undefined" || typeof $("meta[property='al:ios:url']").attr("content") != "undefined") && IS_MOBILE) {
  if(getCookie("appli")!="") open(true);
  else $('#ask_for_app').modal('show');
}
{% endhighlight %}

**In conclusion, you just need to insert the JS script everywhere in the HTML headers. The presence of the applink will do the rest !**


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

The custom URI schemes is a URI with a "custom" protocol, for example `my-app://my-page`.

This protocol becomes active on the mobile phone when the user has installed the corresponding app.

It enables to re-create a sort of "hyperlinks" for mobile apps, as on the web with hypertext. 

It enables to target a precise page in the app. 


#The script

So, let's suppose the script has detected that 

- the applink parameter is present

- the user is viewing the page on a mobile phone

there is a potential redirection to propose to the user : the modal box is opened to the user.

What are the next steps for him ?

It is where it becomes tricky because : 

- on Android phones, it is possible to redirect the user to the mobile app, and if the mobile app is not installed, the user will be automatically redirected to Google Play. This can be done with a simple INTENT action: 

{% highlight javascript %}
		window.location = 'intent:/'+$("meta[property='al:android:url']").attr("content").split(':/')[1]+'#Intent;package='+$("meta[property='al:android:package']").attr("content")+';scheme='+$("meta[property='al:android:url']").attr("content").split(':/')[0]+';launchFlags=268435456;end;';
{% endhighlight %}

that can be proposed under a "Download the app" button. But it's not a very fun button when the user has already installed the app !

- on iPhone, it's possible to do the same with : 

{% highlight javascript %}
		window.location = $("meta[property='al:ios:url']").attr("content");

		setTimeout(function() {

        	// If the user is still here, open the App Store
        	if (!document.webkitHidden) {

          	// Replace the Apple ID following '/id'
          	window.location = 'http://itunes.apple.com/app/id' + $("meta[property='al:ios:app_store_id']").attr("content") ;
        	}
		}, 25);
{% endhighlight %}

If the app is already installed (with its custom URI shemes), it's going to launch the app at the correct page. But if the app is not installed, the user will very shortly see an error popup, and redirected to the AppStore with the `setTimeout function`. Not very good, this popup, but we have no other choice. Same problem also with the "Download the app" button presented to users that have already installed the app.

#Download or open the app ?

To avoid the problem of showing a "Download" button to users that have the app already installed, the mainstream solution is to ask the question at the first time and set a cookie then.

The way to ask it can vary from one site to another : 

- "Already installed ?"

- "Open the app" / "Download the app"

- ...

Here is my full script : 

{% highlight javascript %}
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

      window.location = $("meta[property='al:ios:url']").attr("content");

      setTimeout(function() {

        // If the user is still here, open the App Store
        if (!document.webkitHidden) {

          // Replace the Apple ID following '/id'
          window.location = 'http://itunes.apple.com/app/id' + $("meta[property='al:ios:app_store_id']").attr("content");
        }
      }, 25);


    } else {console.log("fdk"); window.location = 'http://itunes.apple.com/app/id'+$("meta[property='al:ios:app_store_id']").attr("content");}

  } else if (IS_ANDROID) {

    // Instead of using the actual URL scheme, use 'intent://' for better UX
    window.location = 'intent:/'+$("meta[property='al:android:url']").attr("content").split(':/')[1]+'#Intent;package='+$("meta[property='al:android:package']").attr("content")+';scheme='+$("meta[property='al:android:url']").attr("content").split(':/')[0]+';launchFlags=268435456;end;';
  }
}

$("#yes").click(function() { open(true); } );
$("#no").click(function() { open(false); } );
$("#mobile").click(function() { window.location = "http://m.selectionnist.com"+ $("meta[property='al:android:url']").attr("content").split(':/')[1]; } );

if( (typeof $("meta[property='al:android:url']").attr("content") != "undefined" || typeof $("meta[property='al:ios:url']").attr("content") != "undefined") && IS_MOBILE) {
  if(getCookie("appli")!="") open(true);
  else $('#ask_for_app').modal('show');
}

{% endhighlight %}