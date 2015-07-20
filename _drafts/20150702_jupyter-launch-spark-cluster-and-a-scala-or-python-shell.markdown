---
layout: post
title:  "Jupyter configuration to launch a cluster on EC2 and interact in Scala or Python"
date:   2015-07-02 23:00:51
categories: bigdata
---

First prepare the [launch of a Spark cluster on EC2 following my previous article](http://christopher5106.github.io/bigdata/2015/05/28/parse-wikipedia-statistics-and-pages-with-Spark.html) and verify everything works.

Install [Anaconda](http://continuum.io/downloads#all) to manage Python libraries.

Install the latest version of ipython

    conda install ipython

Install [Jupyter-Scala](https://github.com/alexarchambault/jupyter-scala)

    wget https://oss.sonatype.org/content/repositories/snapshots/com/github/alexarchambault/jupyter/jupyter-scala-cli_2.10.5/0.2.0-SNAPSHOT/jupyter-scala_2.10.5-0.2.0-SNAPSHOT.tar.xz
    tar xvf jupyter-scala_2.10.5-0.2.0-SNAPSHOT.tar.xz
    cd jupyter-scala_2.10.5-0.2.0-SNAPSHOT/bin
    ./jupyter-scala


ipython profile create spark

    

Launch Jupyter

    ipython notebook
