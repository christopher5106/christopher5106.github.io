---
layout: post
title:  "Google AppEngine, simply brillant. Tutorial"
date:   2016-05-11 23:00:51
categories: continous deployment
---

AppEngine is simply tremendous and extraordinary. Its web interface is simply beautifully designed, fluid and effective.

AppEngine has been for a while, but since june 2014 it supports Docker technology, the open container standard, and is named ["the flexible environment"](https://cloud.google.com/appengine/docs/flexible/custom-runtimes/).

**AppEngine is the kind of service "fully-managed".**

But it provides also the option to go for more customization : it is the purpose of Google Container Engine, which offers more functions. Google Container Engine is based on an opensource technology (K8s or Kubernetes) that can be installed anywhere.

Google Container Engine could be limitating in some ways. It is based on Google Compute Engine, which provides even more flexibility.

So with Google products, as your needs evolve, you can gain flexibility and use more customizable products :

**AppEngine > Container Engine > Compute Engine**

Just in case you need more.

But do we really need more than what AppEngine provides ? What are the benefits of AppEngine ? Are you sure to know all functions ? Let's see in practice.


# Configuring

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

It might be that you uploaded a unstable version of the Docker, in this case it's time to revert to the previous version :

![]({{ site.url }}/img/appengine_revert.png)

# A/B testing in just one clic

Under the *version section*, it is so easy to set up an A/B testing between two uploaded docker versions :

![]({{ site.url }}/img/appengine_abtesting.png)

Select the versions and choose how much traffic to deliver in percentage on each version :

![]({{ site.url }}/img/appengine_split_traffic.png)

You can divide the traffic between more than 2 different versions.

# Other services

Instead of an URL in `https://PROJECT_ID.appspot.com`, you can add your **custom DNS domains and their SSL certificates**.

AppEngine provides you a **memcache** to share keys between instances, a **blobstore** to store data objects.

Lastly, AppEngine offers a **security scanner** that you can run on a daily, weekly, every-2-week or every-4-week basis, to check a few URL against potential vunerabilities.

Why not shifting ?


**Well done Google!**
