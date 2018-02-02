---
layout: post
title:  "Configure Windows 10 for Ubuntu and server X"
date:   2018-02-02 00:00:51
categories: admin
---

In Windows 10, it is now possible to run Ubuntu Bash shell, without dual boot nor virtual machine, directly using the Windows kernel's new properties. It is named **Windows Subsystem for Linux (WSL)**.

In this tutorial, I'll give you the command to install and use Ubuntu shell on a typical enterprise Windows computer.

# Install Ubuntu Shell

First, in **Settings > Update and security > For developers**, activate **Developer mode**:

![](({{ site.url }}/img/windows_developer_mode.PNG)

Second, in **Settings > Applications > Applications and features**, click on **Programs and features**,

![](({{ site.url }}/img/windows_programs_and_features.PNG)

open **activate or desactivate Windows features** panel:

![](({{ site.url }}/img/windows_features_activation.PNG)

enable the "Windows Subsystem for Linux" optional feature (you can also enable the feature with `Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux` in PowerShell as administrator).

Third reboot.

Last, run `lsxrun /install` in the Windows Command Prompt to install Ubuntu Bash, without requiring the activation of the Windows Store of applications.

You'll find the Ubuntu bash under **Bash** in the Windows Command prompt:

![](({{ site.url }}/img/windows_bash.PNG)

# Install a server X


It is possible to run graphical applications from Ubuntu, for that purpose you need to install [Xming X Server for Windows](https://sourceforge.net/projects/xming/). Now you can run `firefox` in your Ubuntu Bash terminal.

To let Ubuntu bash on Windows 10 run `ssh -X` to get a GUI environment on a remote server, run in a Ubuntu Bash terminal,

    sudo apt install ssh xauth xorg
    sudo vi /etc/ssh/ssh_config

and edit the **ssh_config** file, uncommenting or adding the following lines:

    Host *
        ForwardAgent yes
        ForwardX11 yes
        ForwardX11Trusted yes
        Port 22
        Protocol 2
        GSSAPIDelegateCredentials no
        XauthLocaion /usr/bin/xauth

Now, setting the display, you can access your Ubuntu remote server through the Ubuntu server X on your Windows Ubuntu computer:

    export DISPLAY=localhost:0
    ssh -X ...


**Well done!**
