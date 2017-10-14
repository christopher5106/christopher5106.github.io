---
layout: post
title:  "Python packages and their managers: Ubuntu APT, yum, easy_install, pip, virtualenv, conda"
date:   2017-10-12 00:00:51
categories: python
---

Many of us, computer engineers, might be messed up with Python packages or modules, installing and uninstalling packages with different tools and making it work by chance after many trials.

Indeed, there are many ways to install Python and its modules or packages: the system package manager, but also the multiple python package managers. All these package managers install packages in different directories.

Moreover, the virtual environment managers, as well as the `sudo` command, will demultiply the number of directories in which packages can be found...

After a while, your system might be completely inconsistent, with multiple versions of the same package in different directories, not knowing exactly which one will be used by our programs, and causing errors.

To go directly to the conclusion of this journey, [click here to go to the *Reverse the paths* section](#reverse-the-paths).

### Package managers

To install a Python package, multiple tools are available:

- the system package manager, such as Redhat's `yum` or Ubuntu's `apt-get` commands, can install Python packages:

  ```bash
  sudo apt-get install python python-dev python-all python-all-dev \
  python-numpy python-scipy python-matplotlib python-cycler \
  python-dateutil python-decorator python-joblib python-matplotlib-data \
  python-tz \
  python2.7 python2.7-dev \
  python3 python3-dev python3-numpy python3.5
  ```

    To list the installed packages related to Python:

  ```bash
  apt list --installed | grep python
  # dh-python
  # libboost-mpi-python-dev
  # libboost-mpi-python1.58-dev
  # libboost-mpi-python1.58.0
  # libboost-python-dev
  # libboost-python1.58-dev
  # libboost-python1.58.0
  # libpeas-1.0-0-python3loader
  # libpython-all-dev
  # libpython-dev
  # libpython-stdlib
  # libpython2.7
  # libpython2.7-dev
  # ...
  # python
  # python-all
  # python-all-dev
  # python-apt
  # python-apt-common
  # python-bs4
  # ...
  # python2.7
  # python2.7-dev
  # python2.7-minimal
  # python3
  # python3-apport
  # python3-apt
  # python3-aptdaemon
  # python3-aptdaemon.gtk3widgets
  # python3-aptdaemon.pkcompat
  # python3-blinker
  # python3-botocore
  # python3-brlapi
  # ...
  # python3.5
  # python3.5-dev
  # python3.5-minimal
  ```

    Let's have a look at where the system binaries have been installed:

  ```bash
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

     The system has installed Python packages in the global `dist-packages` directory of each Python version and created symbolic links:

  ```bash
  /usr/lib/python2.7/dist-packages/numpy

  /usr/lib/python3/dist-packages/numpy

  ls -ls /usr/include/numpy
  #-> ../lib/python2.7/dist-packages/numpy/core/include/numpy

  ls -l /usr/include/python2.7/numpy
  #->../../lib/python2.7/dist-packages/numpy/core/include/numpy

  ls -l /usr/include/python3.5m/numpy
  #-> ../../lib/python3/dist-packages/numpy/core/include/numpy
  ```

    Note the good use of `dist-packages` instead of `site-packages` which should be reserved for the system Python.


- `easy_install` in the setupstool package is a Python package manager.

    To install `easy_install` manager on Ubuntu:

      sudo apt-get install python-setuptools python-dev build-essential

    To install a Python package such as Numpy :

  ```bash
  sudo easy_install numpy
  # Searching for numpy
  # Best match: numpy 1.11.0
  # Adding numpy 1.11.0 to easy-install.pth file
  #
  # Using /usr/lib/python2.7/dist-packages
  # Processing dependencies for numpy
  # Finished processing dependencies for numpy
  ```
    In this case, `easy_install` is using `/usr/lib/python2.7/dist-packages` to install the packages, where it has found previously installed Numpy with system `apt-get`. Note the requirement of `sudo` because `/usr/lib/python2.7/dist-packages` is system owned. Let's remove previously installed Numpy and re install it with `easy_install`:

  ```bash
  sudo apt-get remove python-numpy
  sudo easy_install numpy

  ###### compilation of Numpy #######

  # creating /usr/local/lib/python2.7/dist-packages/numpy-1.13.3-py2.7-linux-x86_64.egg
  # Extracting numpy-1.13.3-py2.7-linux-x86_64.egg to /usr/local/lib/python2.7/dist-packages
  # Adding numpy 1.13.3 to easy-install.pth file
  # Installing f2py script to /usr/local/bin
  #
  # Installed /usr/local/lib/python2.7/dist-packages/numpy-1.13.3-py2.7-linux-x86_64.egg
  # Processing dependencies for numpy
  # Finished processing dependencies for numpy
  ```

    `easy_install` has now messed the system directories :).

- `pip` (and `pip3`) is a more recent Python 2 (respectively Python3) package management system which has become a better alternative to `easy_install` for installing Python packages. It is included by default for Python 2 >=2.7.9 or Python 3 >=3.4 , otherwise requires to be installed:

    - with `easy_install`:

          sudo easy_install pip

   - directly with Python with the script [get-pip.py](https://bootstrap.pypa.io/get-pip.py) to run

          sudo python get-pip.py

       In order to avoid to mess up with packages installed by the system, it is possible to specify to pip to install the packages in a local directory rather than globally, with  `--prefix=/usr/local/` option for example.

    - with your sytem package manager

          sudo apt-get install python-pip
          sudo apt-get install python3-pip

    As `easy_install`, `pip` installs packages in `/usr/local/lib/python2.7/dist-packages/`, finding previously installed packages with `apt-get` or `easy_install` and requiring `sudo`:

  ```bash
  sudo pip install numpy
  # Requirement already satisfied: numpy in /usr/local/lib/python2.7/dist-packages/numpy-1.13.3-py2.7-linux-x86_64.egg

  sudo pip uninstall numpy
  # Uninstalling numpy-1.13.3:
  #   /usr/local/lib/python2.7/dist-packages/numpy-1.13.3-py2.7-linux-x86_64.egg
  # Proceed (y/n)? y
  #   Successfully uninstalled numpy-1.13.

  sudo pip install numpy
  # Collecting numpy
  #   Downloading numpy-1.13.3-cp27-cp27mu-manylinux1_x86_64.whl (16.6MB)
  #     100% |████████████████████████████████| 16.7MB 84kB/s
  # Installing collected packages: numpy
  # Successfully installed numpy-1.13.3
  ```

    **It is not recommanded to use `sudo` for a Python package manager other than the system package manager `apt-get`, nor to mess up the system directories.**

    For that purpose, it is recommanded to install the Python2 package in the local user directory `~/.local/` also:

      pip install --user numpy

    In this case, the package can be found

      /home/christopher/.local/lib/python2.7/site-packages/numpy

    Note the change from `dist-packages` to `site-packages` in local mode.

    For `pip3`, the package is direcly installed locally:

      /home/christopher/.local/lib/python3.5/site-packages/numpy

    while for `sudo pip3`, it is installed globally:

      /usr/local/lib/python3.5/dist-packages/numpy

    In recent Ubuntu versions, by default, `pip` installs the package locally.

    To check where your package has been installed:

  ```bash
  pip show numpy
  # Name: numpy
  # Version: 1.13.3
  # Summary: NumPy: array processing for numbers, strings, records, and objects.
  # Home-page: http://www.numpy.org
  # Author: NumPy Developers
  # Author-email: numpy-discussion@python.org
  # License: BSD
  # Location: /home/christopher/miniconda2/lib/python2.7/site-packages
  # Requires
  ```

    To upgrade `pip`:

      pip install -U pip setuptools

    Note it is possible to specify the version to install:

      pip install numpy            # latest version
      pip install numpy==1.9.0     # specific version
      pip install 'numpy>=1.9.0'     # minimum version

    Since `pip` does not have a true depency resolution, you will need to define a requirement file to
    specify which packages and versions needs to be installed and install them:

      pip install -r requirements.txt

    To list installed packages:

      pip list

    To create a requirements file from your installed packages to reproduce your install:

      pip freeze > requirements.txt

    The following command will uninstall the first Python package `pip` has found:

      pip uninstall numpy

    Note that uninstalling Numpy installed by `apt-get` with `pip uninstall` will leave the system in an inconsistent state, with `apt-get` thinking Numpy is in its latest version while it has been uninstalled.

    `pip` offers many other [options and configuration properties](https://pip.pypa.io/en/stable/user_guide).


- `conda` is a package and dependency manager for Python, R, Ruby, Lua, Scala, Java, JavaScript, C/ C++, FORTRAN. It performs a true dependency resolution.

    I would recommand to install Miniconda, which installs `conda`, while Anaconda also installs a hundred packages such as numpy, scipy, ipython notebook, and so on. To install Anaconda from `conda`, simply `conda install anaconda`.

    Its [install under Linux](https://conda.io/docs/user-guide/install/linux.html#install-linux-silent) is very easy. For example:

      wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh
      bash Miniconda2-latest-Linux-x86_64.sh

    It simply creates a `~/miniconda2/` directory. Conda uses the virtual environment mechanism proposed by Python executable to modify its package install path, a mechanism that we'll see later. Uninstalling conda simply consists in removing its directory :

      rm -rf ~/miniconda2

     It proposes to add its binaries to the PATH environment variable by setting it in the `.bashrc` file and here is what happen:

  ```bash
  which python
  # /home/christopher/miniconda2/bin/python

  which pip
  # /home/christopher/miniconda2/bin/pip

  which pip3
  # /usr/bin/pip3

  which easy_install
  # /home/christopher/miniconda2/bin/easy_install

  ls -l /home/christopher/miniconda2/bin/python ls -l /home/christopher/miniconda2/bin/pip /home/christopher/miniconda2/bin/easy_install
  # /home/christopher/miniconda2/bin/easy_install
  # /home/christopher/miniconda2/bin/pip
  # /home/christopher/miniconda2/bin/python -> python2.7
  ```

    Conda install contains `pip` and `easy_install` because some packages are not available under Conda and you'll still need `pip`, which will not simplify your life:

  ```bash
  sudo pip uninstall numpy

  pip install numpy
  #  Collecting numpy
  #    Using cached numpy-1.13.3-cp27-cp27mu-manylinux1_x86_64.whl
  #  Installing collected packages: numpy
  #  Successfully installed numpy-1.13.3

  pip show numpy
  # Name: numpy
  # Version: 1.13.3
  # Summary: NumPy: array processing for numbers, strings, records, and objects.
  # Home-page: http://www.numpy.org
  # Author: NumPy Developers
  # Author-email: numpy-discussion@python.org
  # License: BSD
  # Location: /home/christopher/miniconda2/lib/python2.7/site-packages
  # Requires
  ```

    Default install location for `conda`, `pip` and `easy_install` is now:

      /home/christopher/miniconda2/lib/python2.7/site-packages

    `pip install` has created two subdirectories:

      /home/christopher/miniconda2/lib/python2.7/site-packages/numpy
      /home/christopher/miniconda2/lib/python2.7/site-packages/numpy-1.13.3.dist-info

    This `pip` does not see anymore the packages installed by system package manger `apt-get` nor by system `/usr/bin/pip` but still sees previously installed package in local mode `/usr/bin/pip install --user` while `conda` don't see any of them, many due to a bug in `conda` (because `pip` is consistent with `python` executable).

    To install a package with `conda`:

      conda install numpy

    which overwrite previously installed package by `pip`, and create a new subdirectory:

      /home/christopher/miniconda2/lib/python2.7/site-packages/numpy
      /home/christopher/miniconda2/lib/python2.7/site-packages/numpy-1.13.3-py2.7.egg-info

    Note here the difference with `pip` that would not accept to install a package under the same path.

    `conda list` displays both package references for `conda` and `pip`:

  ```bash
  conda list numpy
  # numpy                     1.13.3           py27hbcc08e0_0  
  # numpy                     1.13.3                    <pip>
  ```

    while `pip` does see the packages installed by `conda` but merges the references in one line:

      ```bash
      conda list numpy
      # numpy                     1.13.3                    <pip>
      ```

    `pip uninstall` is able to uninstall any package it finds in the Python sys path, leaving the other package managers in an inconsistent state (`apt-get`, `conda`,...) while `conda` cannot manage anything else than its own packages.

    It is possible to specify the `pip` packages in the `conda` environment definition to install them as with a pip requirement files:

      # environment.yml
      name: my_app
      dependencies:
      - python>=3.5
      - anaconda
      - numpy
      - pip
      - pip:
        - numpy


    To remove a package from the current environment:

      conda uninstall numpy

    The package 'numpy-1.13.3-py27' still appears in your conda dir `~/miniconda2/pkgs/` but is not available. To clean unused packages:

      conda clean --packages

    'numpy-1.12.1-py36_0' package has not been removed because it is used by another environment.

    As we'll see in the last section, `conda` install creates a clean root environment, using the mechanism of virtual environments. Even with a system in a inconsistent state, with packages installed via the system manager, or different managers and in multiple versions, only newly installed Python packages via `pip` or `conda`, residing in the `~/miniconda2` directory, will effectively be considered by Python programs.

    Nevertheless, be careful: if, before installing `conda`, you had installed via `pip` packages such as Jupyter or iPython packages that setup an executable script (beginning with `#!/usr/bin/python`) in `/usr/local/bin/`, the executables will run under previous packages (before installing `conda`). To have them run the packages of the new root `conda` environment, override them with `conda install jupyter`.

    `conda` also offers the possibility to create new virtual environments.


From this point, you might have a system in an inconsistent state,


### Paths

In the command shell, check which version of Python you're using:

```bash
which python
#/home/christopher/miniconda2/bin/python

python --version
#Python 2.7.14 :: Anaconda, Inc.

which ipython
#/home/christopher/miniconda2/bin/ipython

which jupyter
# /home/christopher/miniconda2/bin/jupyter
```

or in a Python/iPython/Jupyter shell

```python
>>> import sys
>>> sys.executable
'/home/christopher/miniconda2/bin/python'
```

In this case, they look consistent because I've installed iPython and Jupyter with Conda package manager. So, they will use the same packages, which will simplify my development and my install. Ananconda's `python`, `ipython` and `jupyter` binaries come first (before system binaries in /usr/bin) due to the PATH environment variable setup from the Anaconda install:

    export PATH="/home/christopher/miniconda2/bin:$PATH"

Note that Ananconda is using the system binary for Python command as `pip` does:

```bash
ls -l /home/christopher/miniconda2/bin/python
# /home/christopher/miniconda2/bin/python -> python2.7
```


To check where `pip` installs the user packages, run in a Python shell:

```python
>>> import site
>>> site.USER_BASE
'/home/christopher/.local'
>>> site.USER_SITE
'/home/christopher/.local/lib/python2.7/site-packages'
```

To check which directories (and their order of precedence) are used to load the packages / dependencies during a Python run:

```python
Python 2.7.14 |Anaconda, Inc.| (default, Oct  5 2017, 07:26:46)
[GCC 7.2.0] on linux2
Type "help", "copyright", "credits" or "license" for more information.
>>> import sys
>>> sys.path
['', '/home/christopher/technologies/caffe/python', '/home/christopher/apps/christopher5106.github.io', '/home/christopher/miniconda2/lib/python27.zip', '/home/christopher/miniconda2/lib/python2.7', '/home/christopher/miniconda2/lib/python2.7/plat-linux2', '/home/christopher/miniconda2/lib/python2.7/lib-tk', '/home/christopher/miniconda2/lib/python2.7/lib-old', '/home/christopher/miniconda2/lib/python2.7/lib-dynload', '/home/christopher/.local/lib/python2.7/site-packages', '/home/christopher/miniconda2/lib/python2.7/site-packages']
```

**These commands gives a good picture of what's happening when running a Python script, where packages are fetched from and this can be a first step in debugging, although it does not give us any clue on what package has been installed in each directory.**

The presence of the first path in the list is due to the environment variable `PYTHONPATH` setup in my `~/.bashrc` file:

    export PYTHONPATH=~/technologies/caffe/python:$PYTHONPATH

The second path correspond to the local directory in which I have run the `python` command: all local python files are included. These paths are setup at at Python launch, with the contents of any .pth file paths created, and the standard library directories.

Running directly `/usr/bin/python2.7` will launch the default Python setup:

```python
Python 2.7.12 (default, Nov 19 2016, 06:48:10)
[GCC 5.4.0 20160609] on linux2
Type "help", "copyright", "credits" or "license" for more information.
>>> import sys
>>> sys.path
['', '/usr/local/lib/python2.7/dist-packages/subprocess32-3.2.7-py2.7-linux-x86_64.egg', '/usr/local/lib/python2.7/dist-packages/functools32-3.2.3.post2-py2.7.egg', '/home/christopher/technologies/caffe/python', '/home/christopher/apps', '/usr/lib/python2.7', '/usr/lib/python2.7/plat-x86_64-linux-gnu', '/usr/lib/python2.7/lib-tk', '/usr/lib/python2.7/lib-old', '/usr/lib/python2.7/lib-dynload', '/home/christopher/.local/lib/python2.7/site-packages', '/usr/local/lib/python2.7/dist-packages', '/usr/lib/python2.7/dist-packages', '/usr/lib/python2.7/dist-packages/gtk-2.0']
```

It is possible to add any new directory dynamically in the Python code with `sys.path.append("/home/my/path")` or to specify an order of precedence `sys.path.insert(0, "/home/my/path")`.

Last, when loading a package, you can check from which directory it has been loaded from:

```python
>>> import numpy
>>> numpy.__file__
'/home/christopher/miniconda2/lib/python2.7/site-packages/numpy/__init__.pyc'
```

**This command gives a second clue on what package version has been used.**

### Virtual environments

Virtual environments enable packages to be installed locally to an application, and not globally. This enables to have specific version of a package for the application.

`virtualenv` (for pip), `venv` (for pip3), or `conda` are virtual environment managers. I'll create 3 environments, named 'my_app', in each of them. Once the environment 'my_app' is activated, the system command in the bash shell will mention the current active environment "my_app" the following way:

    (my_app) christopher@christopher-G751JY:~/apps$

Let's see for each one:

- `virtualenv`

    To install, either one of the following commands:

      sudo apt get python-virtualenv
      sudo pip install virtualenv
      conda install virtualenv

    Virtual environments are attached to a directory (directory of the application) where packages will be stored.

    To create a virtual environment:

  ```bash
  mkdir my_app
  virtualenv my_app
  # New python executable in /home/christopher/apps/my_app/bin/python
  # Installing setuptools, pip, wheel...done.
  ```

    To activate the environment:

      source my_app/bin/activate

    Let's now list the packages in this environment:

  ```bash
  pip list
  # pip (8.1.1)
  # pkg-resources (0.0.0)
  # setuptools (20.7.0)
  ```
    All other packages installed globally are not visible anymore.

    Note that `pip3` and `conda` package managers will ignore the current virtualenv environment and install packages globally.

    Let's now check the Python executable:
  ```bash
  which python
  # /home/christopher/apps/my_app/bin/python
  ls -l /home/christopher/apps/my_app/bin/python
  #/home/christopher/apps/my_app/bin/python -> python3
  ls -l /home/christopher/apps/my_app/bin/python3
  # /home/christopher/apps/my_app/bin/python3 -> /usr/bin/python3
  ```

    `virtualenv` environment mechanism relies on Python executable ability to configure its environment from local files (if present) and load local packages.

    Inside the environment, paths are modified to:

  ```python
  Python 2.7.12 (default, Nov 19 2016, 06:48:10)
  [GCC 5.4.0 20160609] on linux2
  Type "help", "copyright", "credits" or "license" for more information.
  >>> import sys
  >>> sys.path
  ['', '/home/christopher/technologies/caffe/python', '/home/christopher/apps', '/home/christopher/apps/my_app/lib/python2.7', '/home/christopher/apps/my_app/lib/python2.7/plat-x86_64-linux-gnu', '/home/christopher/apps/my_app/lib/python2.7/lib-tk', '/home/christopher/apps/my_app/lib/python2.7/lib-old', '/home/christopher/apps/my_app/lib/python2.7/lib-dynload', '/usr/lib/python2.7', '/usr/lib/python2.7/plat-x86_64-linux-gnu', '/usr/lib/python2.7/lib-tk', '/home/christopher/apps/my_app/local/lib/python2.7/site-packages', '/home/christopher/apps/my_app/lib/python2.7/site-packages']
  >>> import numpy
  Traceback (most recent call last):
    File "<stdin>", line 1, in <module>
  ImportError: No module named numpy
  ```

    Packages installed globally by `apt-get` are not visible anymore. This is consistent with `pip` behavior inside the environment.

    Install path for packages in this environment with `pip install` are now:

      /home/christopher/apps/my_app/lib/python2.7/site-packages


- `venv`

    To install:

      sudo apt-get install python3-venv

    To create a virtual environment:

      mkdir my_app
      python3 -m venv my_app

    To activate the environment:

      source my_app/bin/activate

    Let's now check the Python executable:
  ```bash
  which python
  # /home/christopher/apps/my_app/bin/python
  ls -l /home/christopher/apps/my_app/bin/python
  #/home/christopher/apps/my_app/bin/python -> python3
  ls -l /home/christopher/apps/my_app/bin/python3
  # /home/christopher/apps/my_app/bin/python3 -> /usr/bin/python3
  ```

    `venv` environment mechanism relies on Python executable ability to configure its environment from local files (if present) and load local packages.

    Inside the environment, paths are modified to:

  ```python
  Python 3.5.2 (default, Sep 14 2017, 22:51:06)
  [GCC 5.4.0 20160609] on linux
  Type "help", "copyright", "credits" or "license" for more information.
  >>> import sys
  >>> sys.path
  ['', '/home/christopher/technologies/caffe/python', '/home/christopher/apps', '/usr/lib/python35.zip', '/usr/lib/python3.5', '/usr/lib/python3.5/plat-x86_64-linux-gnu', '/usr/lib/python3.5/lib-dynload', '/home/christopher/apps/my_app/lib/python3.5/site-packages']
  >>> import numpy
  Traceback (most recent call last):
    File "<stdin>", line 1, in <module>
  ImportError: No module named numpy
  ```

    Behavior is the same as for `virtualenv`.

- `conda`

    Conda uses Python executable ability to configure its environment from a local directory. In conda, packages and environments are stored directly in `~/miniconda2/`.

    To create an environment:

      conda create --name my_app

    Note that it is possible to specify the version of Python:

      conda create --name my_app python=3.4

    or to create an environment from a file:

      conda env create --file environment.yml

    To activate the environment:

      source activate my_app

    Let's now check the Python executable:

  ```bash
  which python
  # /home/christopher/miniconda2/bin/python
  ```

    `conda` environment mechanism relies on configurations and libraries defined in the `conda` directory to load the packages relative to the environment.

    Inside the environment, paths are modified to:

  ```python
  Python 2.7.13 |Anaconda, Inc.| (default, Sep 30 2017, 18:12:43)
  [GCC 7.2.0] on linux2
  Type "help", "copyright", "credits" or "license" for more information.
  >>> import sys
  >>> sys.path
  ['', '/home/christopher/technologies/caffe/python', '/home/christopher/apps', '/home/christopher/miniconda2/lib/python27.zip', '/home/christopher/miniconda2/lib/python2.7', '/home/christopher/miniconda2/lib/python2.7/plat-linux2', '/home/christopher/miniconda2/lib/python2.7/lib-tk', '/home/christopher/miniconda2/lib/python2.7/lib-old', '/home/christopher/miniconda2/lib/python2.7/lib-dynload', '/home/christopher/.local/lib/python2.7/site-packages', '/home/christopher/miniconda2/lib/python2.7/site-packages']
  >>> import numpy
  Traceback (most recent call last):
    File "<stdin>", line 1, in <module>
  ImportError: No module named numpy
  ```

    As we have already seen, a `conda` virtual environment does not contain global packages, or previous installed packages prior to `conda` install, except those installed with `pip install --user`, prior or after.

    `pip` still installs Python packages in

      ~/miniconda2/lib/python2.7/site-packages

    so, packages installed via `pip` are global to all `conda` environments. The same is true for packages installed via `conda` in the default/root environment.

    Packages installed via `conda` in the local environment are found in:

      ~/miniconda2/envs/my_app/lib/python2.7/site-packages/

    While globally installed `ipython` or `jupyter` executables will not use the local environment's packages, let's try to see what happens when running `conda install jupyter` in the environment:

  ```bash
  which ipython
  # /home/christopher/miniconda2/envs/my_app/bin/ipython

  which jupyter
  # /home/christopher/miniconda2/envs/my_app/bin/jupyter
  ```

    Each executable has been deployed in the local environment's bin folder, and begins with the local Python command `#!/home/christopher/miniconda2/envs/my_app/bin/python`. So, now you can run `ipython` or `jupyter` in your application, using the packages required by the application.

    To deactivate an environment:

      source deactivate my_app

    To delete an environment:

      conda env remove -n my_app


### Reverse the paths

To get a view on all versions of a package installed in all virtual environments, user rights, and package managers:

    sudo find / -name "numpy*" 2>/dev/null

**This is our last clue...**

So, here is the reverse meaning of each path you might encounter:


    /usr/local/lib/python2.7/dist-packages/numpy

means Numpy has been installed with `easy_install`

.

    /usr/lib/python2.7/dist-packages/numpy

means Numpy has been installed by either
- `apt-get` system manager with packet `python-numpy`
- `easy_install` or `pip` in `sudo` mode for a system default Python 2

.

    /usr/lib/python3/dist-packages/numpy

means Numpy has been installed by `apt-get` system manager with packet `python3-numpy`

.

    /usr/local/lib/python3.5/dist-packages/numpy

means Numpy has been installed by `sudo pip3 install`

.

    /home/christopher/.local/lib/python2.7/site-packages/numpy

means Numpy has been installed with `pip install --user`, even in the case if `conda` is installed on the system

.

    /home/christopher/.local/lib/python3.5/site-packages/numpy

means Numpy has been installed with `pip3 install` with a Python 3 using user directory by default

.

    /home/christopher/apps/my_app/lib/python2.7/site-packages/numpy

means either:
- a specific install path has been specified to a package manager
- Numpy has been installed in a `virtualenv` environment

.


    /home/christopher/miniconda2/lib/python2.7/site-packages/numpy
    /home/christopher/miniconda2/lib/python2.7/site-packages/numpy-1.13.3-py2.7.egg-info

means
- `conda` has been installed on the system
- Numpy has been installed in conda root/default environment with `conda`

.


    /home/christopher/miniconda2/lib/python2.7/site-packages/numpy
    /home/christopher/miniconda2/lib/python2.7/site-packages/numpy-1.13.3.dist-info

means
- `conda` has been installed on the system
- Numpy has been installed in conda root/default environment with `pip` from conda install

.

    /home/christopher/miniconda2/pkgs/numpy-1.13.3-py27hbcc08e0_0/lib/python2.7/site-packages/numpy

means
- `conda` has been installed on the system
- Numpy has been installed with `conda`

but does not mean that Numpy is being used in the current environment or any other environments

.

    /home/christopher/miniconda2/envs/my_app/lib/python2.7/site-packages/numpy
    /home/christopher/miniconda2/envs/yad2k/lib/python3.6/site-packages/numpy

means Numpy has been installed by `conda` package manager in two `conda` environments, 'my_app' and 'yad2k', each one using a different version of Python.


### In conclusion

`conda` looks like a far better tool to manage Python packages, but since `pip` is still required for some packages, it comes with several problems:

- some packages are not seen by `conda`, the ones installed previously via `pip install --user`. I would recommand to remove Python packages in your `~/.local` directory and to **never use `pip install --user`**

- in the meantime, take the opportunity to never ever use `sudo` to install any package once `conda` has been installed on your system

- packages installed via `pip` are global to all `conda` environments, so they do not benefit from `conda` separation of packages

- packages installed via `conda` in the default/root environment are also global to all `conda` environments, but `conda list` does not reference them

- some Python executables installed prio to `conda` setup might use the previous Python environment, in particular the previous packages.

- last, environment variable PYTHONPATH is still active in `conda` environment. Cross directory references should be avoided when using  virtual environments because they are not consistent with the behavior of virtual environments.


It is far from simple...

**Well done!**
