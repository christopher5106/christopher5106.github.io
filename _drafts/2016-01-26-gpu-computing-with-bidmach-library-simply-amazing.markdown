---
layout: post
title:  "BIDMach library for GPU computing with Intel parallel studio XE: simply amazing [install on MacOS and EC2]"
date:   2016-01-22 23:00:51
categories: parallel computing
---


Using Intel Parallel Studio,

<iframe width="560" height="315" src="https://www.youtube.com/embed/G6pKqD8uXdk" frameborder="0" allowfullscreen></iframe>


there is a very great [BIDMach library](http://bid2.berkeley.edu/bid-data-project/download/) with [very competitive results(see the benchmarks)](https://github.com/BIDData/BIDMach/wiki/Benchmarks).

# Install on Mac OS

As usual, it requires an iMac with a NVIDIA GPU and its CUDA library installed.

Since I'm using CUDA 7.5 instead of 7.0, I had to recompile JCuda, BIDMat, and BIDMach.

First download Intel parallel studio.

    mkdir ~/technologies/JCuda
    cd ~/technologies/JCuda
    git clone https://github.com/jcuda/jcuda-common.git
    git clone https://github.com/jcuda/jcuda-main.git
    git clone https://github.com/jcuda/jcuda.git
    git clone https://github.com/jcuda/jcublas.git
    git clone https://github.com/jcuda/jcufft.git
    git clone https://github.com/jcuda/jcusparse.git
    git clone https://github.com/jcuda/jcurand.git
    git clone https://github.com/jcuda/jcusolver.git
    cmake jcuda-main
    make
    cd jcuda-main
    mvn install

    git clone https://github.com/BIDData/BIDMach.git

    #compiling for GPU
    export PATH=$PATH:/usr/local/cuda/bin/
    cd ~/technologies/BIDMach
    cd jni/src
    ./configure
    make
    make install
    cd ../..

    #compiling for CPU
    cd src/main/C/newparse
    ./configure
    make
    make install
    cd ../../../..

    ./getdevlibs.sh
    rm lib/IScala.jar
    cp ../JCuda/jcuda-main/target/* lib/
    rm lib/jcu*0.7.0a.jar
    cp ../BIDMat/lib/libbidmatcuda-apple-x86_64.dylib lib/
    sbt compile
    sbt package
    ./bidmach

which gives :

    Loading /Users/christopher5106/technologies/BIDMach/lib/bidmach_init.scala...
    import BIDMat.{CMat, CSMat, DMat, Dict, FMat, FND, GMat, GDMat, GIMat, GLMat, GSMat, GSDMat, GND, HMat, IDict, Image, IMat, LMat, Mat, SMat, SBMat, SDMat}
    import BIDMat.MatFunctions._
    import BIDMat.SciFunctions._
    import BIDMat.Solvers._
    import BIDMat.Plotting._
    import BIDMach.Learner
    import BIDMach.models.{Click, FM, GLM, KMeans, KMeansw, LDA, LDAgibbs, Model, NMF, SFA, RandomForest, SVD}
    import BIDMach.networks.DNN
    import BIDMach.datasources.{DataSource, MatSource, FileSource, SFileSource}
    import BIDMach.datasinks.{DataSink, MatSink}
    import BIDMach.mixins.{CosineSim, Perplexity, Top, L1Regularizer, L2Regularizer}
    import BIDMach.updaters.{ADAGrad, Batch, BatchNorm, IncMult, IncNorm, Telescoping}
    import BIDMach.causal.IPTW
    1 CUDA device found, CUDA version 7.5

    Welcome to Scala version 2.11.2 (Java HotSpot(TM) 64-Bit Server VM, Java 1.8.0_51).
    Type in expressions to have them evaluated.
    Type :help for more information.


Everything works well, my GPU is found correctly.

    ./scripts/getdata.sh

# EC2 launch

create a EC2 security group, for example bidmach ,in zone us-west-1b

a keypair us-west2-keypair

    dry-run :
    aws ec2 run-instances --dry-run --image-id ami-71280941 --key-name us-west2-keypair --security-groups bidmach --instance-type g2.2xlarge --placement AvailabilityZone=us-west-2b --region us-west-2


    aws ec2 run-instances --image-id ami-71280941 --key-name us-west2-keypair --security-groups bidmach --instance-type g2.2xlarge --placement AvailabilityZone=us-west-2b --region us-west-2


    aws ec2 terminate-instances --instance-ids i-3178b5f6
