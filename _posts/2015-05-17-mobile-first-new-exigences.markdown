---
layout: post
title:  "New mobile exigences for web sites"
date:   2015-05_17 23:00:51
categories: mobile
---

Since Google has changed its rank computation to prioritize web sites that are correctly designed for mobiles,

[page speed for m.selectionnist.com](https://developers.google.com/speed/pagespeed/insights/?url=m.selectionnist.com)

it's time to change the paradigm.

Webapp design was before that event mostly driven on a design pattern that was similar to the IOS and Android apps, that is :
- all the code for the "view" in the client,  
- asynchronous request to the server for the data (Ajax request in case of a webapp)

This kind of design is not possible anymore :

- CSS from frameworks are too heavy to load for Google Page Speed, **all the content above the fold should be available in only 2 HTTP round trips".**

- Javascript is also too heavy, in particular when including other libraries, even when minified and compressed.

- If we use non-blocking Javascript, Javascript will also require a second HTTP call to request the data to display.

So, the only way to do that, is that the server delivers the first data for the URL it is requested and the javascript will take over when loaded.

Eventually, we have to go one step further in the **full-stack full JS pattern**. It is time for a unique code on **server-side and client-side**.

Such technologies to do this are :

- [Rendr](http://rendrjs.github.io/) for Backbone

- [Angular-Server](https://www.npmjs.com/package/angularjs-server) for AngularJS

- [FastBoot](https://github.com/tildeio/ember-cli-fastboot) for EmberJS

- and Facebook framework, [React](https://github.com/mhart/react-server-example)

This practices requires to **stop using jQuery** because there is no DOM on server-side.
