---
layout: post
title:  "Commands for NVIDIA install on Ubuntu 16.04"
date:   2016-12-30 00:00:51
categories: nvidia
---

# Remove the kernel you don't need

Check your boot partition :

    df -h
    /dev/sda1       236M  224M     0 100% /boot

OMG.

Get your current kernel :

    uname -r
    4.4.0-43-generic

List installed kernels :

    dpkg --list 'linux-image*'

Remove some of them :

    sudo apt-get remove  linux-image-4.4.0-21-generic linux-image-4.4.0-45-generic linux-image-4.4.0-47-generic linux-image-4.4.0-51-generic

NB : if your partition /boot is full and your install broken, it might ask you to run `apt-get -f install` which might not work as well, due to space constraints. You can erase manually the kernels you do not need :

    rm /boot/vmlinuz-4.4.0-42-generic


Remove also your NVIDIA drivers :

    sudo apt-get purge nvidia*

    root@s3:~# dpkg --list 'nvidia*'
    Desired=Unknown/Install/Remove/Purge/Hold
    | Status=Not/Inst/Conf-files/Unpacked/halF-conf/Half-inst/trig-aWait/Trig-pend
    |/ Err?=(none)/Reinst-required (Status,Err: uppercase=bad)
    ||/ Name                                  Version                 Architecture            Description
    +++-=====================================-=======================-=======================-================================================================================
    un  nvidia-legacy-340xx-vdpau-driver      <none>                  <none>                  (no description available)
    un  nvidia-libopencl1-dev                 <none>                  <none>                  (no description available)
    un  nvidia-vdpau-driver                   <none>                  <none>                  (no description available)


Remove the packages that you don't need anymore :

    sudo apt-get autoremove


Check your boot partition :

    df -h
    /dev/sda1       236M   97M  127M  44% /boot


Update :

    sudo apt-get update && sudo apt-get -y upgrade


Install kernel extras if not already installed :

    sudo apt-get install -y linux-image-extra-`uname -r`

and reboot

    sudo reboot


# Install NVIDIA drivers from NVIDIA


Download and install latest drivers :

    wget http://us.download.nvidia.com/XFree86/Linux-x86_64/375.20/NVIDIA-Linux-x86_64-375.20.run
    chmod +x NVIDIA-Linux-x86_64-375.20.run
    sudo ./NVIDIA-Linux-x86_64-375.20.run

The first bash command is :

```
nvidia-smi
```

If the command returns

    Failed to initialize NVML: Driver/library version mismatch

try again the install procedure.



# Install NVIDIA drivers from Ubuntu


You can install

    sudo apt install ubuntu-drivers-common

to list the devices

    ubuntu-drivers devices

    root@s3:~# ubuntu-drivers devices
    == /sys/devices/pci0000:80/0000:80:03.0/0000:85:00.0/0000:86:10.0/0000:88:00.0 ==
    model    : GK210GL [Tesla K80]
    vendor   : NVIDIA Corporation
    modalias : pci:v000010DEd0000102Dsv000010DEsd0000106Cbc03sc02i00
    driver   : nvidia-367 - third-party free
    driver   : nvidia-370 - third-party free
    driver   : xserver-xorg-video-nouveau - distro free builtin
    driver   : nvidia-375 - third-party free recommended

    == cpu-microcode.py ==
    driver   : intel-microcode - distro non-free


and either install automatically

    sudo ubuntu-drivers autoinstall

or automatically

    sudo apt-get install nvidia-375


# Install Cuda 8

Check your Ubuntu version

    lsb_release -a
    No LSB modules are available.
    Distributor ID:	Ubuntu
    Description:	Ubuntu 16.04.1 LTS
    Release:	16.04
    Codename:	xenial

Install CUDA 8 for Ubuntu 16.04 :

    wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.44-1_amd64.deb
    sudo dpkg -i cuda-repo-ubuntu1604_8.0.44-1_amd64.deb
    sudo apt-get update
    sudo apt-get install cuda
