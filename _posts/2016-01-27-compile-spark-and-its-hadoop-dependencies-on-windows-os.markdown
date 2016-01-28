---
layout: post
title:  "Compile Spark on Windows"
date:   2016-01-27 23:00:51
categories: big data
---

Spark has become the main big data tool, very easy to use as well as very powerful. Built on top of some Hadoop classes, Spark offers the use of the distributed memory (RDD) as if you were working on a single machine, and 3 REPL shells `spark-shell`, `pyspark` and `sparkR` for their respective Scala, Python and R languages. It is possible to submit a script with `spark-submit` command and to develop or test locally with `--master local[1]` option, before launching on a [cluster of hundred of instances such as EMR](http://christopher5106.github.io/big/data/2016/01/19/computation-power-as-you-need-with-EMR-auto-termination-cluster-example-random-forest-python.html).

Here I recompile Spark on Windows since it avoids problems one could encounter with Windows binaries, such as software version mismatches. Loading and compiling all the required dependencies on a slow network and with standard hardware may require a day or so. The steps are the following :

- Download and install [Java Development Kit 7](http://www.oracle.com/technetwork/java/javase/downloads/jdk7-downloads-1880260.html) in a path such as **C:\Java** (it has to be a folder with spaces)

- Download and install [Python 2.7.11](https://www.python.org/downloads/).

- Add **C:\Python27\;C:\Java** to your `Path` environment variable, **C:\Java** to your `JAVA_HOME` env var.

Check everything works well :

    where java
    >C:\Java\bin\java.exe
    >C:\Windows\System32\java.exe

our install arrives first

    java -version
    >java version "1.7.0_79"
    >Java(TM) SE Runtime Environment (build 1.7.0_79-b15)
    >Java Hotspot(TM) 64-Bit Server VM (build 24.79-b02, mixed mode)

Java should be 64bit to increase memory above 2G. Try javac

    javac

I also had to change the memory options to `-Xmx2048m` (instead of 516m) in **C:\Program Files (x86)\sbt\conf\sbtconfig.txt**.

    python --version
    > Python 2.7.11

- Download and compile Spark :

        git clone git://github.com/apache/spark.git
        sbt package
        sbt assembly

that will create the Spark assembly JAR.

Once created, launch Pypark:

    bin\pyspark --master local[1]
