---
layout: post
title:  "Python packages and their managers: Ubuntu APT, yum, easy_install, pip, virtualenv, conda"
date:   2017-10-12 00:00:51
categories: python
---

Many of us might be messed up with Python packages or modules.

There are many ways to install Python and its modules or packages: system package manager, python package managers, ... These package managers install packages in different directories.

Moreover, the virtual environment managers, as well as the `sudo` command, will demultiply the number of directories in which packages can be found...

After a while, your system might be completely inconsistent.

### Package managers

- the system package manager, such as Redhat's `yum` or Ubuntu's `apt-get` commands to install Python packages:

  ```bash
  sudo apt-get install python python-dev python-all python-all-dev
  python-numpy python-scipy python-matplotlib python-cycler
  python-dateutil python-decorator python-joblib python-matplotlib-data
  python-tz
  python2.7 python2.7-dev python3 python3-dev python3-numpy python3.5
  ```

    To list the installed packages:

  ```bash
  apt list --installed | grep python

  dh-python/xenial-updates,xenial-updates,now 2.20151103ubuntu1.1 all  [installé]
  libboost-mpi-python-dev/xenial,now 1.58.0.1ubuntu1 amd64  [installé, automatique]
  libboost-mpi-python1.58-dev/xenial-updates,now 1.58.0+dfsg-5ubuntu3.1 amd64  [installé, automatique]
  libboost-mpi-python1.58.0/xenial-updates,now 1.58.0+dfsg-5ubuntu3.1 amd64  [installé, automatique]
  libboost-python-dev/xenial,now 1.58.0.1ubuntu1 amd64  [installé]
  libboost-python1.58-dev/xenial-updates,now 1.58.0+dfsg-5ubuntu3.1 amd64  [installé, automatique]
  libboost-python1.58.0/xenial-updates,now 1.58.0+dfsg-5ubuntu3.1 amd64  [installé, automatique]
  libpeas-1.0-0-python3loader/xenial,now 1.16.0-1ubuntu2 amd64  [installé, automatique]
  libpython-all-dev/xenial,now 2.7.11-1 amd64  [installé, automatique]
  libpython-dev/xenial,now 2.7.11-1 amd64  [installé, automatique]
  libpython-stdlib/xenial,now 2.7.11-1 amd64  [installé, automatique]
  libpython2.7/xenial-updates,xenial-security,now 2.7.12-1ubuntu0~16.04.1 amd64  [installé]
  libpython2.7-dev/xenial-updates,xenial-security,now 2.7.12-1ubuntu0~16.04.1 amd64  [installé, automatique]
  ...
  python/xenial,now 2.7.11-1 amd64  [installé]
  python-all/xenial,now 2.7.11-1 amd64  [installé, automatique]
  python-all-dev/xenial,now 2.7.11-1 amd64  [installé]
  python-apt/xenial,now 1.1.0~beta1build1 amd64  [installé, automatique]
  python-apt-common/xenial,xenial,now 1.1.0~beta1build1 all  [installé, automatique]
  python-bs4/xenial,xenial,now 4.4.1-1 all  [installé, automatique]
  ...
  python2.7/xenial-updates,xenial-security,now 2.7.12-1ubuntu0~16.04.1 amd64  [installé, automatique]
  python2.7-dev/xenial-updates,xenial-security,now 2.7.12-1ubuntu0~16.04.1 amd64  [installé, automatique]
  python2.7-minimal/xenial-updates,xenial-security,now 2.7.12-1ubuntu0~16.04.1 amd64  [installé, automatique]
  python3/xenial,now 3.5.1-3 amd64  [installé]
  python3-apport/xenial-updates,xenial-updates,xenial-security,xenial-security,now 2.20.1-0ubuntu2.10 all  [installé, automatique]
  python3-apt/xenial,now 1.1.0~beta1build1 amd64  [installé, automatique]
  python3-aptdaemon/xenial,xenial,now 1.1.1+bzr982-0ubuntu14 all  [installé, automatique]
  python3-aptdaemon.gtk3widgets/xenial,xenial,now 1.1.1+bzr982-0ubuntu14 all  [installé, automatique]
  python3-aptdaemon.pkcompat/xenial,xenial,now 1.1.1+bzr982-0ubuntu14 all  [installé, automatique]
  python3-blinker/xenial,xenial,now 1.3.dfsg2-1build1 all  [installé, automatique]
  python3-botocore/xenial-updates,xenial-updates,now 1.4.70-1~16.04.0 all  [installé, automatique]
  python3-brlapi/xenial-updates,now 5.3.1-2ubuntu2.1 amd64  [installé, automatique]
  ...
  python3.5/xenial-security,now 3.5.2-2ubuntu0~16.04.1 amd64 [installed,upgradable to: 3.5.2-2ubuntu0~16.04.3]
  python3.5-dev/xenial-security,now 3.5.2-2ubuntu0~16.04.1 amd64 [installed,upgradable to: 3.5.2-2ubuntu0~16.04.3]
  python3.5-minimal/xenial-security,now 3.5.2-2ubuntu0~16.04.1 amd64 [installed,upgradable to: 3.5.2-2ubuntu0~16.04.3]
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

     The system has installed Python packages in the global `dist-packages` directories and created symbolic links:

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



- `easy_install` in the setupstool package is a Python package manager

      sudo apt-get install python-setuptools python-dev build-essential

    To install the Pandas package :

      easy_install pandas


- `pip` (and `pip3`) is more recent Python 2 (respectively Python3) package management system. It is included by default for Python 2 >=2.7.9 or Python 3 >=3.4 , otherwise requires to be installed:

    - with easy_install:

          easy_install pip

        To install the Pandas package:

          pip install numpy

        The newly installed package can be found:

          /usr/local/lib/python2.7/dist-packages/numpy

        To install it the package in the local user directory `~/.local/` :

          pip install --user numpy

        In this case, the package can be found

          /home/christopher/.local/lib/python2.7/site-packages/numpy

        Note the change from `dist-packages` to `site-packages` in local mode.

   - directly with Python with the script [get-pip.py](https://bootstrap.pypa.io/get-pip.py) to run

          python get-pip.py

       In order to avoid to mess up with packages installed by the system, it is possible to specify to pip to install the packages in a local directory rather than globally, with  `--prefix=/usr/local/` option for example.

    - with your sytem package manager

          sudo apt-get install python-pip
          sudo apt-get install python3-pip

        In recent Ubuntu versions, by default, `pip` installs the package locally:

          /home/christopher/.local/lib/python3.5/site-packages/numpy


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

    `pip` has become a better alternative to `easy_install` for installing Python packages.

    Note it is possible to specify the version to install:

      pip install numpy            # latest version
      pip install numpy==1.9.0     # specific version
      pip install 'numpy>=1.9.0'     # minimum version

    Since Pip does not have a true depency resolution, you will need to define a requirement file to
    specify which package needs to be installed and install them:

      pip install -r requirements.txt

    To list installed packages:

      pip list

    To create a requirements file from your installed packages to reproduce your install:

      pip freeze > requirements.txt

    To uninstall a package:

      pip uninstall numpy

    Pip offers many other [options and configuration properties](https://pip.pypa.io/en/stable/user_guide).


- `conda` is a package and dependency manager for Python, R, Ruby, Lua, Scala, Java, JavaScript, C/ C++, FORTRAN. It performs a true dependency resolution.

    I would recommand to install Miniconda, which installs `conda`, while Anaconda also installs a hundred packages such as numpy, scipy, ipython notebook, and so on. To install Anaconda from `conda`, simply `conda install anaconda`.

    Its [install under Linux](https://conda.io/docs/user-guide/install/linux.html#install-linux-silent) is very easy and simply creates a `~/miniconda2/` directory and adds its binaries to the PATH environment variable by setting it in the `.bashrc` file.

    Uninstalling conda simply consists in removing its directory :

      rm -rf ~/miniconda2

    To install a package:

      conda install numpy

    To check installed packages:

  ```bash
  conda list numpy
  # numpy                     1.13.3           py27hbcc08e0_0  
  # numpy                     1.13.3                    <pip>
  ```

    Here, numpy package has been installed at least twice. Once with `conda`, once with `pip`. Note:

    - `pip` does not see the packages installed by the system as well as the packages installed via conda

    - `conda` does not see the packages installed by the system

    Note that since `conda` sees the `pip` packages, it is possible to specify the `pip` packages in the `conda`
    listing the packages:

      # environment.yml
      name: my_app
      dependencies:
      - python>=3.5
      - anaconda
      - numpy
      - pip
      - pip:
        - numpy

    The most recent package 'numpy-1.13.3-py27' has been installed in `~/miniconda2/pkgs/`, although I had a less recent package 'numpy-1.12.1-py36_0' for another environment.

    To remove a package from the current environment:

      conda uninstall numpy

    The package 'numpy-1.13.3-py27' still appears in your conda dir `~/miniconda2/pkgs/` but is not available. To clean unused packages:

      conda clean --packages

    'numpy-1.12.1-py36_0' package has not been removed because it is used by another environment.

    As we'll see in the last section, `conda` also offers a virtual environment manager.


To conclude:

- **I would recommand to never use `sudo` to run the  `pip` and `conda` Python package managers. Reserve `sudo` for the system packages `apt-get`.** Packages installed with `sudo` will not be removed by `pip uninstall` command.

- From this point, you should begin to leave your system in an inconsistent state. Packages installed with system manager, or different managers begin to be messed up: multiple versions of a same package can be fetched by the Python programs, and we do not know which one.


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
      pip install virtualenv
      conda install virtualenv

    Virtual environments are attached to a directory (directory of the application) where packages will be stored.

    To create a virtual environment:

      mkdir my_app
      virtualenv my_app

    To activate the environment:

      source my_app/bin/activate

    Let's now list the packages in this environment:

  ```bash
  pip list
  pip (8.1.1)
  pkg-resources (0.0.0)
  setuptools (20.7.0)
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

- `venv`

    To install:

      sudo apt-get install python3-venv

    To create a virtual environment:

      mkdir my_app
      python3 -m venv my_app

    To activate the environment:

      source my_app/bin/activate

    Note that `conda` package manager will ignore the current virtualenv environment and install packages globally, while `pip` will not.

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

- `conda`

    In conda, environments are not linked to a particular directory. They are stored directly in `~/miniconda2/envs/`.

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

    Note that as surprising as it looks like, `conda list` does not see anymore the packages that have been installed with `pip` and `pip3` globally, while they are still active and appear with `pip list` (or `pip3 list`)

    To deactivate an environment:

      source activate my_app

    To delete an environment:

      conda env remove -n my_app


To get a view on all versions of a package installed in all virtual environments, user rights, and managers:

    sudo find / -name "numpy" 2>/dev/null

**This is our last clue command...**
