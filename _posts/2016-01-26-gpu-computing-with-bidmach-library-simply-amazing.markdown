---
layout: post
title:  "BIDMach library for GPU computing with Intel parallel studio XE: simply amazing [install on MacOS and EC2]"
date:   2016-01-26 23:00:51
categories: parallel computing
---


Using Intel Parallel Studio,

<iframe width="560" height="315" src="https://www.youtube.com/embed/G6pKqD8uXdk" frameborder="0" allowfullscreen></iframe>


there is a very great [BIDMach library](http://bid2.berkeley.edu/bid-data-project/download/) with [very competitive results(see the benchmarks)](https://github.com/BIDData/BIDMach/wiki/Benchmarks): on some tasks, **one can achieve on a single GPU instance the speed of a cluster of a few hundred instances, at a cost 10 to 1000 times lower**.


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

    ./bidmach
    val a = loadSMat("data/rcv1/docs.smat.lz4")

returns

    a: BIDMat.SMat =
    (   33,    0)    1
    (   47,    0)    1
    (   94,    0)    1
    (  104,    0)    1
    (  112,    0)    3
    (  118,    0)    1
    (  141,    0)    2
    (  165,    0)    2
       ...   ...   ...

Let's continue on the [Quickstart tutorial](https://github.com/BIDData/BIDMach/wiki/Quickstart) :

    val c = loadFMat("data/rcv1/cats.fmat.lz4")
    val (mm, mopts) = GLM.learner(a, c, 1)
    mm.train

To clear the cache :

    resetGPU; Mat.clearCaches

# EC2 launch

BIDMach team has compiled an EC2 AMI, available on the US west zone (Oregon).

First, add an EC2 permission policy to your user :

    {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "StmtXXX",
                "Effect": "Allow",
                "Action": [
                    "ec2:DescribeAvailabilityZones",
                    "ec2:RunInstances",
                    "ec2:TerminateInstances",
                    "ec2:CreateSecurityGroup",
                    "ec2:CreateKeyPair",
                    "ec2:DescribeInstances"
                ],
                "Resource": [
                    "*"
                ]
            }
        ]
    }

in order to create a EC2 security group `bidmach` and a keypair `us-west2-keypair` and start the instance, all in zone us-west-2 where the AMI lives :

    aws ec2 create-security-group --group-name bidmach --description bidmach --region us-west-2
    aws ec2 create-key-pair --key-name us-west2-keypair --region us-west-2
    # Save the keypair to us-west2-keypair.pem and change its mode
    sudo chmod 600 us-west2-keypair.pem
    aws ec2 run-instances --image-id ami-71280941 --key-name us-west2-keypair --security-groups bidmach --instance-type g2.2xlarge --placement AvailabilityZone=us-west-2b --region us-west-2

Get your instance public DNS with

    aws ec2 describe-instances --region us-west-2

Connect to the instance  :

    ssh -i us-west2-keypair.pem ec2-user@ec2-XXX_DNS.us-west-2.compute.amazonaws.com

Let's download the data :

    /opt/BIDMach/scripts/getdata.sh
    /opt/BIDMach/bidmach


Start BIDMach with `bidmach` command and you get :

    Loading /opt/BIDMach/lib/bidmach_init.scala...
    import BIDMat.{CMat, CSMat, DMat, Dict, FMat, FND, GMat, GDMat, GIMat, GLMat, GSMat, GSDMat, HMat, IDict, Image, IMat, LMat, Mat, SMat, SBMat, SDMat}
    import BIDMat.MatFunctions._
    import BIDMat.SciFunctions._
    import BIDMat.Solvers._
    import BIDMat.Plotting._
    import BIDMach.Learner
    import BIDMach.models.{DNN, FM, GLM, KMeans, KMeansw, LDA, LDAgibbs, Model, NMF, SFA, RandomForest}
    import BIDMach.datasources.{DataSource, MatDS, FilesDS, SFilesDS}
    import BIDMach.mixins.{CosineSim, Perplexity, Top, L1Regularizer, L2Regularizer}
    import BIDMach.updaters.{ADAGrad, Batch, BatchNorm, IncMult, IncNorm, Telescoping}
    import BIDMach.causal.IPTW
    1 CUDA device found, CUDA version 6.5


Data should be available in **/opt/BIDMach/data/**. Let's load the data, partition it between train and test, train the model, predict on the test set and compute the accuracy :

    val a = loadSMat("/opt/BIDMach/data/rcv1/docs.smat.lz4")
    val c = loadFMat("/opt/BIDMach/data/rcv1/cats.fmat.lz4")
    val inds = randperm(a.ncols)
    val atest = a(?, inds(0->100000))
    val atrain = a(?, inds(100000->a.ncols))
    val ctest = c(?, inds(0->100000))
    val ctrain = c(?, inds(100000->a.ncols))
    val cx = zeros(ctest.nrows, ctest.ncols)
    val (mm, mopts, nn, nopts) = GLM.learner(atrain, ctrain, atest, cx, 1)
    mm.train
    nn.predict
    val p = ctest *@ cx + (1 - ctest) *@ (1 - cx)
    mean(p, 2)



Stop the instance :

    aws ec2 terminate-instances --instance-ids i-XXX
