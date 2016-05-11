---
layout: post
title:  "Google AppEngine, simply brilliant. Here is my tutorial"
date:   2016-05-11 19:00:51
categories: continous deployment
---

**AppEngine** is simply tremendous and extraordinary. Its web interface is simply beautifully designed, fluid and effective.

AppEngine has been existing for a while, but since june 2014 it supports Docker technology, the open container standard, under the name of ["custom runtimes in the flexible environment"](https://cloud.google.com/appengine/docs/flexible/custom-runtimes/), which offers the ability to do almost anything **... in a very "fully-managed" way.** Much further than the traditional meaning of "fully-managed service" : once you've created your Docker, you do not have to think anymore about deployment :)

AppEngine is today still in *beta*, but it reminds me of Gmail which has been in *beta* for a long while also, and AppEngine might convert everybody switch to it in the near future as did Gmail with mailboxes. While still in *beta*, AppEngine operates on the VM in the US regions, which will soon change.

Google Cloud provides also the option to go for more customization : it is the purpose of **Google Container Engine**, which extends AppEngine with more functions, less management, more customization of the containers, services, instances, ...

If Google Container Engine could be limitating in some ways, Google Container Engine is based

- on an opensource technology (K8s or **Kubernetes**) that can be installed anywhere, in particular on local or on-premise servers that can become the providers of the same interface as Google Container Engine.

- on **Google Compute Engine**, the naked VM, that can be useful in a big project for the use of very specific technologies.

So with Google Cloud, as your needs evolve, you can gain in flexibility, from full management to complete manual setup of public or private instances, each offer based on the more customizable one :

  **AppEngine > Container Engine > Compute Engine or Kubernetes on-premise**

- to begin simple

- in case you need more later on, you're not stuck by some technical limitations.

- to stay independant of Google future, not to stick to AppEngine API only.

But do we really need more than what AppEngine provides ? What are the benefits of AppEngine ? Are you sure to know all about its functions ? Let's have a look.


# Configuring in one command line

After having installed the Gcloud SDK, simply create a project in the [Google Cloud console](https://console.cloud.google.com) and configure your SDK:

    gcloud init

You can re-run this command later to change your configuration (switch from one project to another one for example).

# Publishing in one command line

Create your Docker, with the `EXPOSE 8080` command to expose port 8080 that will be used by AppEngine. Once your **Dockerfile** is ready and has been tested locally, publishing it on AppEngine is just creating a **app.yaml** file :

```yaml
runtime: custom
vm: true
automatic_scaling:
  min_num_instances: 1
```

in the same directory as the Dockerfile and running :

    gcloud preview app deploy app.yaml

Once the deployment finished, your app will be available under the URL `https://PROJECT_ID.appspot.com`.

You can re-run this command later to publish a newer version and see the version number simply incrementing under the "default service" under *Services section* :

![AppEngine services]({{site.url}}/img/appengine_versions.png)

In my case, I submitted 5 times. Under *Versions section* I get the same but with the history :

![AppEngine versions]({{site.url}}/img/appengine_versions.png)


The first two parameters of the previous **app.yaml** file are necessary to be in the flexible environment case, but almost anything is possible with such a configuration file which defines whatever we need, such as the [autoscaling](https://cloud.google.com/appengine/docs/python/config/appref#scaling_elements). In my case of a simple demo, I had no need for redondancy and wanted to have only one instance rather than 2 to reduce my costs. It still provides me the autoscaling in case of a higher traffic :

![autoscaling with one instance in iddle times]({{site.url}}/img/appengine_instances.png)

# Revert to a previous version in just one clic

It might be that you uploaded a unstable version of the Docker, in this case it's time to revert to the previous version by selecting it under the *Versions section* and migrating traffic to it :

![]({{ site.url }}/img/appengine_revert.png)

# A/B testing in just one clic

You would have never imagine how easy this could have been. Under the *Versions section*, click on the "Split Traffic" button to set up an A/B testing between two uploaded docker versions :

![]({{ site.url }}/img/appengine_abtesting.png)

Select the versions and choose how much traffic to deliver in percentage on each version :

![]({{ site.url }}/img/appengine_split_traffic.png)

You can divide the traffic between more than 2 different versions.

# Other

You can add more than one service, forward ports, ... by configuring correctly the **

Instead of an URL in `https://PROJECT_ID.appspot.com`, you can add your **custom DNS domains and their SSL certificates**.

AppEngine provides you a **memcache** to share keys between instances, a **blobstore** to store data objects.

Lastly, AppEngine offers a **security scanner** that you can run on a daily, weekly, every-2-week or every-4-week basis, to check a few URL against potential vunerabilities.

Why not shifting ?


**Well done Google!**
