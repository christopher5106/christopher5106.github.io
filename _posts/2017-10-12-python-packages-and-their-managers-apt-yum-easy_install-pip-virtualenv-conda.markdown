---
layout: post
title:  "Python packages and their managers: Ubuntu APT, yum, easy_install, pip, virtualenv, conda"
date:   2017-10-12 00:00:51
categories: python
---

Many of us might be messed up with Python packages or modules.

There are many ways to install Python and its modules or packages:

- with the package managers: system package manager, python package managers, ...

- with virtual environment managers

### Package managers

- the system package manager, such as Redhat's `yum` or Ubuntu's `apt-get` commands:

  ```bash
  sudo apt-get install python python-dev python-all python-all-dev
  python-numpy python-scipy python-matplotlib python-cycler
  python-dateutil python-decorator python-joblib python-matplotlib-data
  python-tz
  python2.7 python2.7-dev python3 python3-dev python3-numpy python3.5

  ls -l /usr/bin/*python*
  # /usr/bin/dh_python2
  # /usr/bin/dh_python3 -> ../share/dh-python/dh_python3
  # /usr/bin/dh_python3-ply
  # /usr/bin/python -> python2.7
  # /usr/bin/python2 -> python2.7
  # /usr/bin/python2.7
  # /usr/bin/python2.7-config -> x86_64-linux-gnu-python2.7-config
  # /usr/bin/python2-config -> python2.7-config
  # /usr/bin/python3 -> python3.5
  # /usr/bin/python3.5
  # /usr/bin/python3.5-config -> x86_64-linux-gnu-python3.5-config
  # /usr/bin/python3.5m
  # /usr/bin/python3.5m-config -> x86_64-linux-gnu-python3.5m-config
  # /usr/bin/python3-config -> python3.5-config
  # /usr/bin/python3m -> python3.5m
  # /usr/bin/python3m-config -> python3.5m-config
  # /usr/bin/python-config -> python2.7-config
  # /usr/bin/x86_64-linux-gnu-python2.7-config
  # /usr/bin/x86_64-linux-gnu-python3.5-config -> x86_64-linux-gnu-python3.5m-config
  # /usr/bin/x86_64-linux-gnu-python3.5m-config
  # /usr/bin/x86_64-linux-gnu-python3-config -> x86_64-linux-gnu-python3.5-config
  # /usr/bin/x86_64-linux-gnu-python3m-config -> x86_64-linux-gnu-python3.5m-config
  # /usr/bin/x86_64-linux-gnu-python-config -> x86_64-linux-gnu-python2.7-config
    ```


    The `python` command is a shortcurt to `python2.7`, and `python3` command is a shortcurt to `python3.5`. Versions 2 and 3 of Python are quite different and still coexist in the Python ecosystem.

    `python-config` (a shortcut to `x86_64-linux-gnu-python2.7-config`) outputs build options for python C/C++ extensions or embedding.

    `dh_python2` calculates Python dependencies, adds maintainer scripts to byte compile files, etc.

     `python3-ply` is the Lex and Yacc implementation for Python3 and `dh_python3-ply` generates versioned dependencies on `python3-ply`


- `easy_install` in the setupstool package is a Python package manager

      sudo apt-get install python-setuptools python-dev build-essential

    To install the Pandas package :

      easy_install pandas


- `pip` (and `pip3`) is more recent Python 2 (respectively Python3) package management system. It is included by default for Python 2 >=2.7.9 or Python 3 >=3.4 , otherwise requires to be installed:

    - with easy_install:

          easy_install pip

        To install the Pandas package:

          pip install Pandas

        To install it the package in the local user directory `~/.local/` :

          pip install --user Pandas


   - directly with Python with the script [get-pip.py](https://bootstrap.pypa.io/get-pip.py) to run

          python get-pip.py

       In order to avoid to mess up with packages installed by the system, it is possible to specify to pip to install the packages in a local directory rather than globally, with  `--prefix=/usr/local/` option for example.

    - with your sytem package manager

          sudo apt-get install python-pip
          sudo apt-get install python3-pip

        In recent Ubuntu versions, by default, `pip` installs the package locally.


    To upgrade `pip`:

      pip install -U pip setuptools

    `pip` has become a better alternative to `easy_install` for installing Python packages.

    Note it is possible to specify the version to install:

      pip install SomePackage            # latest version
      pip install SomePackage==1.0.4     # specific version
      pip install 'SomePackage>=1.0.4'     # minimum version

    Since Pip does not have a true depency resolution, you will need to define a requirement file to
    specify which package needs to be installed and install them:

        pip install -r requirements.txt

    To list installed packages:

        pip list

    To create a requirements file from your installed packages to reproduce your install:

        pip freeze > requirements.txt

    Pip offers many other [options and configuration properties](https://pip.pypa.io/en/stable/user_guide).




**I would recommand to never use `sudo` to run the Python package manager. Reserve `sudo` for the system packages.**

From this point, you should begin to leave your system in an inconsistent state.






### Virtual environments

- `virtualenv`

    sudo pip install --upgrade virtualenv

    pip install virtualenv
    virtualenv <DIR>
    source <DIR>/bin/activate


- `conda`



### Paths

In the command shell, check which version of Python you're using:

```bash
which python
#/home/christopher/miniconda2/bin/python
python --version
#Python 2.7.14 :: Anaconda, Inc.
```

or in a Python shell

```python
>>> import sys
>>> sys.executable
'/home/christopher/miniconda2/bin/python'
```

To check where pip installs the user packages, run in a Python shell:

```python
>>> import site
>>> site.USER_BASE
'/home/christopher/.local'
>>> site.USER_SITE
'/home/christopher/.local/lib/python2.7/site-packages'
```
