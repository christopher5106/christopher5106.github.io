---
layout: post
title:  "Optimize your revenues and earn money with your blog : tools for the webmaster"
date:   2015-10-19 23:00:51
categories: marketing
---

**The key important thing is to write articles about which visitors could be interested about.** Sounds obvious, but all the strategy then consists of choosing the right content, with the right words, corresponding to the keywords people look for in search engines.

Let's see the different tools to help you in your earnings optimization.

# Track Analytics with Google Analytics

Google Analytics is the leading platform to track your visitors, and view reports on your analytics :

![png]({{ site.url }}/img/google_analytics.png)

After 10 articles, I got 1700 users per month. It's time to monetize it.

# Advertising with Google Adsense

Add advertising to your website with [Google Adsense](http://www.google.com/adsense)

Create an ad unit and add it to your website.

Add [multiple add units](https://support.google.com/adsense/answer/187698?hl=en&ref_topic=2717009) to the page to optimize your revenues. I usually insert once at the beginning of the HTML layout :

{% highlight javascript %}
<script async src="//pagead2.googlesyndication.com/pagead/js/adsbygoogle.js"></script>
{% endhighlight %}

and at the end of the HTML layout as discussed [here](http://stackoverflow.com/questions/25095912/how-do-you-use-multiple-adsense-units-on-one-page)

{% highlight html %}
<script>
  // (adsbygoogle = window.adsbygoogle || []).push({});
  [].forEach.call(document.querySelectorAll('.adsbygoogle'), function(){
    (adsbygoogle = window.adsbygoogle || []).push({});
  });
</script>
{% endhighlight %}

Then, I can add the provided snippet

{% highlight html %}
<ins class="adsbygoogle"
     style="display:block"
     data-ad-client="XXX"
     data-ad-slot="YYY"
     data-ad-format="auto"></ins>
{% endhighlight %}

wherever I wand to place an add, in between the two previous snippets of code.

Add AdSense in your Google Analytics panel via [AdSense Linking in Google Analytics](https://support.google.com/adsense/answer/6084409?hl=en) in order to see AdSense metrics in Analytics reports, see earnings and ad impressions based on user visits and pages.

# Improve your web site interface and ergonomy with Page Analytics

[Page Analytics](https://chrome.google.com/webstore/detail/page-analytics-by-google/fnbdnhhicmebfgdgglcdacdapkcihcoh) is a Chrome Extension to see conversions / click rates directly while navigating on the web pages for which you have Google Analytics enabled :

![Page analytics]({{ site.url }}/img/page_analytics.png)


# Improve your search presence with Google Webmaster Tools

[Google Webmaster Tools](http://www.google.com/webmasters/tools) has a search console to help you check your site, such as the presence of a sitemap, the display in search results... and will email you if any unusual event occurs, such a drop discontinuity for your search rank.

![png]({{ site.url }}/img/search_presence.png)

Do not forget to link your search console in Google analytics

![link search console]({{ site.url }}/img/link_search_console.png)

![link search console]({{ site.url }}/img/link_search_console2.png)

# Optimize your SEO with Woorank


[Woorank](https://www.woorank.com) helps you oversee all aspects of search engine optimization, with a report including most important aspects.


# Check your position in keyword searches with Positeo

[Positeo](http://www.positeo.com/check-position/) helps you check your position in search results for some keywords.


# Check keyword revenues with Adwords Keyword planner

[Keyword Planner](https://adwords.google.com/KeywordPlanner)

![png]({{ site.url }}/img/keyword_tool.png)

This tool can help you decide which subject and keywords to optimize your revenues. In this example, the keyword "avion" will bring 0,23€ per click, while the keyword "deep learning" brings 1,5€ per click.

This analysis can help you define your subjects, which words to use in your document, in order to maximize your revenues.


# Improve your page rank thanks to links from other websites

To improve your visibility in search results, the best is to get links from other websites with high page ranks. You can check the page ranks on [Page Rank websites](http://www.pagerank.fr).

Getting links from websites of page rank 4 is a good start.

Media websites have usually page ranks above 8, such as  [Le Monde journal's page rank](http://www.pagerank.fr/rapport-indexation.fr.html?uri=www.lemonde.fr).
