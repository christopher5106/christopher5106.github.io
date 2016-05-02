---
layout: post
title:  "Deploying with Docker and Kubernetes - tutorial from your PC to AWS EC2, Google cloud or any private servers"
date:   2016-05-02 23:00:51
categories: continous deployment
---

You might have read my first post about [deployment with Chef technology](http://christopher5106.github.io/continous/deployment/2015/03/17/deployment-from-your-pc-to-your-cloud-best-practice.html) one year ago.

1 year later, things have changed a bit, become easier, in particular with the arrival of the opensource technology **Kubernetes** from Google.

# The needs for deployment

The needs are still the same :

- Deploy applications quickly and predictably
- Scale applications
- Deploy continuously new features (*continuous deployment*)
- Optimize cost
- Be independent : be able to deploy on multiple providers

This is the last point I will show in practice in this tutorial.

The main problematic is the lack of support from main providers, the absence of warranties, and changes in specifications without consultation (modification or removal of some specifications) which lead you to adapt to all the changes of their platforms - quite unevitable.

The solution against technological dependency might be

- the "oakham razor" principle : build specifically for what you need, without the superfluous. Performance, and work efficency will be better, but in case of an unprecise need, you'll finish at end having re-implemented every functionality of some market solutions,

- the "standard" principle, meaning to use a technology if it is widely adopted by a **real** sufficient number of customers,

- the "open-source" principle, where the community will provide lot's of advantages such as the wide adoption, the reliability, the number of features and extensions, the availability of developers and support, and a long term life cycle.

# The shift in methodology

Old methods use **dedicated servers**

- *Data privacy with on-promise servers*

- *Performance guarantees*

Current methods use **virtual servers** : the build occurs during deployment, with deployment scripts, and update scripts, inside the virtual machine, and for all applications at the same time. These methods, brought by technologies such as Chef, Puppet or Ansible, have the benefits of *scalability* and *reproducibility*.

![]({{ site.url }}/img/current_deployment_methods.png)

Docker and deployment platform technologies have enabled new deployment methods, that add new benefits to the previous ones :

- builds occur during development time (not deployment) which leads to a better *consistency* between dev and prod stages

- apps are separated into different containers, which implies *loose coupling* and *modularity*

- deployment supervised by infrastructure is more *reliable* than with scripts and agents inside the VM

- monitoring of containers is easier than VM

- the slow process of launching a VM occurs only once, and deploying an app means in this case managing containers, which leads to better *agility*

- containers can be deployed on different provider, this is *portability*

![]({{ site.url }}/img/new_deployment_methods.png)

Each container can have its own IP.

Let's see in practice how to deploy on a cluster on your machine, and on clusters of VM in the AWS EC2 and Google Cloud.

# Create an app and package it in a container

Let's create a very simple nodeJS app as explained in [Kubernetes Hellonode example](http://kubernetes.io/docs/hellonode/) with one file **server.js** :

```javascript
var http = require('http');
var handleRequest = function(request, response) {
  response.writeHead(200);
  response.end("Hello World!");
}
var www = http.createServer(handleRequest);
www.listen(8080);
```

and a **Dockerfile** :

```bash
FROM node:0.12
EXPOSE 8080
COPY server.js .
CMD node server.js
```

and build it (require to have Docker installed) :

```bash
# start the default Docker VM
docker-machine start

# Google : use the project ID in your cloud console
docker build -t gcr.io/PROJECT_ID/hello-node:v1 .
```


# Launch the cluster of VM



# Scale the number of replicas

```bash
kubectl scale deployment hello-node --replicas=4
```
