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
        python-dateutil python-decorator python-joblib python-matplotlib-data python-tz
        python2.7 python2.7-dev python3 python3-dev python3-numpy python3.5

        ls -l /usr/bin/*python*

    returns

        -rwxr-xr-x 1 root root    1056 déc.  10  2015 /usr/bin/dh_python2
        lrwxrwxrwx 1 root root      29 mai   18  2016 /usr/bin/dh_python3 -> ../share/dh-python/dh_python3
        -rwxr-xr-x 1 root root    2336 oct.  16  2014 /usr/bin/dh_python3-ply
        lrwxrwxrwx 1 root root       9 juil.  4  2016 /usr/bin/python -> python2.7
        lrwxrwxrwx 1 root root       9 juil.  4  2016 /usr/bin/python2 -> python2.7
        -rwxr-xr-x 1 root root 3546104 nov.  19  2016 /usr/bin/python2.7
        lrwxrwxrwx 1 root root      33 nov.  19  2016 /usr/bin/python2.7-config -> x86_64-linux-gnu-python2.7-config
        lrwxrwxrwx 1 root root      16 déc.  10  2015 /usr/bin/python2-config -> python2.7-config
        lrwxrwxrwx 1 root root       9 juil.  4  2016 /usr/bin/python3 -> python3.5
        -rwxr-xr-x 2 root root 4460336 nov.  17  2016 /usr/bin/python3.5
        lrwxrwxrwx 1 root root      33 nov.  17  2016 /usr/bin/python3.5-config -> x86_64-linux-gnu-python3.5-config
        -rwxr-xr-x 2 root root 4460336 nov.  17  2016 /usr/bin/python3.5m
        lrwxrwxrwx 1 root root      34 nov.  17  2016 /usr/bin/python3.5m-config -> x86_64-linux-gnu-python3.5m-config
        lrwxrwxrwx 1 root root      16 mars  23  2016 /usr/bin/python3-config -> python3.5-config
        lrwxrwxrwx 1 root root      10 juil.  4  2016 /usr/bin/python3m -> python3.5m
        lrwxrwxrwx 1 root root      17 mars  23  2016 /usr/bin/python3m-config -> python3.5m-config
        lrwxrwxrwx 1 root root      16 déc.  10  2015 /usr/bin/python-config -> python2.7-config
        -rwxr-xr-x 1 root root    2909 nov.  19  2016 /usr/bin/x86_64-linux-gnu-python2.7-config
        lrwxrwxrwx 1 root root      34 nov.  17  2016 /usr/bin/x86_64-linux-gnu-python3.5-config -> x86_64-linux-gnu-python3.5m-config
        -rwxr-xr-x 1 root root    3185 nov.  17  2016 /usr/bin/x86_64-linux-gnu-python3.5m-config
        lrwxrwxrwx 1 root root      33 mars  23  2016 /usr/bin/x86_64-linux-gnu-python3-config -> x86_64-linux-gnu-python3.5-config
        lrwxrwxrwx 1 root root      34 mars  23  2016 /usr/bin/x86_64-linux-gnu-python3m-config -> x86_64-linux-gnu-python3.5m-config
        lrwxrwxrwx 1 root root      33 déc.  10  2015 /usr/bin/x86_64-linux-gnu-python-config -> x86_64-linux-gnu-python2.7-config


-
