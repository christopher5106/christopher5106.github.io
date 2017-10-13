---
layout: post
title:  "Python packages and their managers: Ubuntu APT, pip, virtualenv, conda"
date:   2017-10-12 00:00:51
categories: python
---

Many of us might be messed up with Python packages or modules.

There are many ways to install Python and its modules or packages:

- the system package manager, such as `apt-get` for Ubuntu

        sudo apt-get install python python-dev python-all python-all-dev
        python-numpy python-scipy python-matplotlib python-cycler
        python-dateutil python-decorator python-joblib python-matplotlib-data
        python-tz
        python2.7 python2.7-dev python3 python3-dev python3-numpy python3.5

        ls -l /usr/bin/*python*

    returns

        /usr/bin/dh_python2
        /usr/bin/dh_python3 -> ../share/dh-python/dh_python3
        /usr/bin/dh_python3-ply
        /usr/bin/python -> python2.7
        /usr/bin/python2 -> python2.7
        /usr/bin/python2.7
        /usr/bin/python2.7-config -> x86_64-linux-gnu-python2.7-config
        /usr/bin/python2-config -> python2.7-config
        /usr/bin/python3 -> python3.5
        /usr/bin/python3.5
        /usr/bin/python3.5-config -> x86_64-linux-gnu-python3.5-config
        /usr/bin/python3.5m
        /usr/bin/python3.5m-config -> x86_64-linux-gnu-python3.5m-config
        /usr/bin/python3-config -> python3.5-config
        /usr/bin/python3m -> python3.5m
        /usr/bin/python3m-config -> python3.5m-config
        /usr/bin/python-config -> python2.7-config
        /usr/bin/x86_64-linux-gnu-python2.7-config
        /usr/bin/x86_64-linux-gnu-python3.5-config -> x86_64-linux-gnu-python3.5m-config
        /usr/bin/x86_64-linux-gnu-python3.5m-config
        /usr/bin/x86_64-linux-gnu-python3-config -> x86_64-linux-gnu-python3.5-config
        /usr/bin/x86_64-linux-gnu-python3m-config -> x86_64-linux-gnu-python3.5m-config
        /usr/bin/x86_64-linux-gnu-python-config -> x86_64-linux-gnu-python2.7-config


-
