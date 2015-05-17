---
layout: post
title:  "New mobile exigences for web sites"
date:   2015-05-17 23:00:51
categories: mobile
---

Since Google has changed its rank computation to prioritize web sites designed for mobiles :

[example of a 100% page speed for m.selectionnist.com](https://developers.google.com/speed/pagespeed/insights/?url=m.selectionnist.com)

it's time to change the paradigm.

Webapp design was before that event mostly driven on a design pattern that was similar to the IOS and Android app development, that is :

- all the code for the "view" is in the client app,  
- asynchronous requests are made to the server to fetch the data (Ajax request in case of a webapp)

This kind of design pattern, that was also the case in hybrid apps, is not possible anymore since :

- CSS from frameworks are too heavy to load, **all the content above the fold should be available in only 2 HTTP round trips"** (Google Page Speed).

- Javascript is also too heavy, in particular when including other libraries, even when minified and compressed.

- If we use non-blocking Javascript, that will be loaded later once the HTML is loaded, it will also require for the webapp a second HTTP call to request the data to display.

So, the only way to do that, is :

 **the server has to deliver the first data in the HTML for the URL it is requested, and the javascript, loaded asynchronously, will take over for navigation**.

Eventually, it's time to go one step further in the **full-stack full JS** (a unique language for front-end and back-end development) and the **hybrid apps patterns** (a unique code to program downloalable apps - IOS, Android - and webmobile apps) with this last requirement :

**a unique code for server-side and client-side programming**.

Such technologies to fullfill this requirement are based on NodeJS :

- [Rendr](http://rendrjs.github.io/) for Backbone

- [Angular-Server](https://www.npmjs.com/package/angularjs-server) for AngularJS

- [FastBoot](https://github.com/tildeio/ember-cli-fastboot) for EmberJS

- and Facebook framework, [React](https://github.com/mhart/react-server-example)

A first best practices required to enable this is :

- **to stop using jQuery** because there is no DOM on server-side and jQuery code cannot be rendered on server-side.

**We are impatient to see the next improvements in that field in the near future.**
