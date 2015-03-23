---
layout: post
title:  "Enforcing security with Virtual Private Clouds and Security Groups"
date:   2015-03-23 23:00:51
categories: network
---

Here is my practice.

If you want to follow my [advice]({{ site.url }}/continous/deployment/2015/03/17/deployment-from-your-pc-to-your-cloud-best-practice.html), you will create two stacks, `Production` and `Preproduction`. The production stack runs the machines for the production site, whereas the preproduction stack is the exact copy of the production stack in order to check that deployment works before deploying to production.

Before creating the stacks, in order to enforce separation, create two VPC with for example the following network addresses :

- Production `10.0.0.0/16`

- Preproduction `11.0.0.0/16`

Usually I leave **DNS resolution and hostname** to **yes**, because it's quite difficult to work without them.

Under each VPC, I create two private subnets and two public subnets, in two different availability zones, but in the same region (`eu-west-1` in my case).

The two public subnets (`10.0.0.0/24` and `10.0.2.0/24` - with an even number for the third byte) will be used for the loadbalancers and the NAT instance that will be directly on the Internet.
Loadbalancers require two public subnets in order to assure redundancy, in case one zone is not available due to failure.

All other instances will be launched in the two private subnets (`10.0.1.0/24` and `10.0.3.0/24`  - with an odd number for the third byte), not accessible directly from the web, and with no direct access to the web either. This security will protect them from unintended network activities.
I create two private subnets because services, such as RDS instances, require two subnets in order to assure redundancy, in case one zone is not available due to failure.

####Public subnets

Instances will be directly on Internet,

- able to query any Internet service directly,

- being query-able from anywhere on the Internet.

having an IP address and the standard internet gateway in their route table. For this to work :

- Set **Auto-assign Public IP** to true

- Set the route table to

  Destination  | Target
  ------------- | -------------
  10.0.0.0/16  | local
  0.0.0.0/0  | *igw-\*, the internet gateway*


####Private subnets

Security will be stronger : instances will have no IP, so it will not be possible to access them directly from the Internet and also for them to access the Internet directly.

- Set **Auto-assign Public IP** to false

- Launch a NAT instance in the public subnet with a security group I'll name `NATSG`, and an IP address, that will enable filtered communications between private instances and the Internet.

- Set the route table to


  Destination  | Target
  ------------- | -------------
  10.0.0.0/16  | local
  0.0.0.0/0  | *nat instance*

- Open ports for some TCP connections on the security group `NATSG` of the NAT instance that will filter the access of the private instance to web.

    **Inbound ports**



    Type | Port | Source
    ------------- | ------------- | -------------
    MYSQL  | 3306 | 10.0.0.0/16
    GITHUB | 9418 | 10.0.0.0/16
    SMTP | 2525 | 0.0.0.0/0
    SMTPS | 465 | 10.0.0.0/16
    SSH | 22 | 0.0.0.0/0
    HTTP | 80 | 10.0.0.0/16
    HTTPS | 443 | 10.0.0.0/16
    SMTP | 25 | 0.0.0.0/0


    Open these ports for the private instance to access services outside the private networks, such as Github, SMTP servers, download mirrors, etc.



    **Outbound ports**

    For the connection to come back.

    Type | Port | Destination
    ------------- | ------------- | -------------
    HTTP | 80 | 0.0.0.0/0
    HTTPS | 443 | 0.0.0.0/0


Here is the final diagram (thanks to @AWS)
![Chef Workflow](http://docs.aws.amazon.com/AmazonVPC/latest/UserGuide/images/nat-instance-diagram.png)


**Now you're ready to launch your loadbalancers, instances and create your stacks.**
