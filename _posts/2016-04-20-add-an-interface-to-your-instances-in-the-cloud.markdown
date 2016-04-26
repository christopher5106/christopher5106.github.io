---
layout: post
title:  "Add an interface to Ubuntu instances"
date:   2016-04-20 10:00:51
categories: cloud
---

![VNC client]({{ site.url }}/img/vnc_viewer_mac.png)


On your EC2 instance, if the OS is Ubuntu,

```
sudo apt-get install ubuntu-desktop
sudo apt-get install vnc4server
sudo apt-get install gnome-panel
vncserver
```

Add the password.

Kill the process to modify the configuration:

```
vncserver -kill :1
vi .vnc/xstartup
```

Uncomment or add the following lines :

```
unset SESSION_MANAGER
gnome-session -session=gnome-classic & gnome-panel&
```

Launch VNC again :

```
vncserver
```

In the inbound configuration panel of the security group of your EC2 instance, add TCP 5901.

Download a VNC viewer on your computer, such as [realvnc vnc viewer](https://www.realvnc.com/download/viewer/).

Choose the right IP and port :

![]({{ site.url }}/img/vnc_viewer.png)

To secure the connection, create a SSH tunnel :

    ssh -f USER@SERVER-IP -L 8443:localhost:5901 -N

and connect at your tunnel :

![]({{ site.url }}/img/vnc_viewer_secure.png)    

On Windows, you can also create the tunnel with PuTTY :

![]({{ site.url }}/img/vnc_viewer_secure_putty.png)  

An alternative VNC client is TightVNC :


![]({{ site.url }}/img/tightvnc_viewer.png)

**You're done!**
