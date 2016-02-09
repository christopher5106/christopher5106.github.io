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

**Having the library installed locally** presents the great advantage to develop and test on small datasets directly on the local computer, before renting an GPU-enabled instance in the cloud.

As usual, it requires an iMac with a NVIDIA GPU and its CUDA library installed.

Since I'm using CUDA 7.5 instead of 7.0, I had to recompile JCuda, BIDMat, and BIDMach.

First download Intel parallel studio.

{% highlight bash %}
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
{% endhighlight %}

In *bidmach* file, change the CUDA version to the current one `JCUDA_VERSION="0.7.5"` and start `./bidmach` command which gives :

{% highlight scala %}
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
{% endhighlight %}


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

{% highlight scala %}
val c = loadFMat("data/rcv1/cats.fmat.lz4")
val (mm, mopts) = GLM.learner(a, c, 1)
mm.train
{% endhighlight %}

To clear the cache :

{% highlight scala %}
resetGPU; Mat.clearCaches
{% endhighlight %}

# EC2 launch

BIDMach team has compiled an EC2 AMI, available on the US west zone (Oregon).

First, add an EC2 permission policy to your user :

{% highlight json %}
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
{% endhighlight %}

in order to create a EC2 security group `bidmach` and a keypair `us-west2-keypair` and start the instance, all in zone us-west-2 where the AMI lives :

{% highlight bash %}
aws ec2 create-security-group --group-name bidmach --description bidmach \
--region us-west-2

aws ec2 create-key-pair --key-name us-west2-keypair --region us-west-2
# Save the keypair to us-west2-keypair.pem and change its mode
sudo chmod 600 us-west2-keypair.pem

aws ec2 run-instances --image-id ami-71280941 --key-name us-west2-keypair \
--security-groups bidmach --instance-type g2.2xlarge \
--placement AvailabilityZone=us-west-2b --region us-west-2
{% endhighlight %}

Get your instance public DNS with

{% highlight bash %}
aws ec2 describe-instances --region us-west-2
{% endhighlight %}

Connect to the instance  :

{% highlight bash %}
ssh -i us-west2-keypair.pem ec2-user@ec2-XXX_DNS.us-west-2.compute.amazonaws.com
{% endhighlight %}

Let's download the data :

{% highlight bash %}
/opt/BIDMach/scripts/getdata.sh
/opt/BIDMach/bidmach
{% endhighlight %}

Start BIDMach with `bidmach` command and you get :

{% highlight scala %}
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
{% endhighlight %}

Data should be available in **/opt/BIDMach/data/**. Let's load the data, partition it between train and test, train the model, predict on the test set and compute the accuracy :

{% highlight scala %}
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
{% endhighlight %}

During training, you get


- the percentage of consumed train data,
- the negative log likelyhood,
- the gigaflops,
- the times,
- the consumed data gigabytes,
- the megabytes per seconds, and
- the occupied GPU memory

as here :


    corpus perplexity=14737,915077
    Predicting
    3,00%, ll=-0,00783, gf=9,558, secs=0,0, GB=0,00, MB/s=436,75, GPUmem=0,70
    6,00%, ll=-0,00806, gf=9,610, secs=0,0, GB=0,01, MB/s=439,78, GPUmem=0,70
    10,00%, ll=-0,00804, gf=10,101, secs=0,0, GB=0,01, MB/s=462,40, GPUmem=0,70
    13,00%, ll=-0,00802, gf=10,380, secs=0,0, GB=0,01, MB/s=475,39, GPUmem=0,70
    16,00%, ll=-0,00813, gf=10,550, secs=0,0, GB=0,02, MB/s=483,29, GPUmem=0,70
    20,00%, ll=-0,00804, gf=10,605, secs=0,0, GB=0,02, MB/s=485,10, GPUmem=0,70
    23,00%, ll=-0,00793, gf=10,444, secs=0,0, GB=0,02, MB/s=477,68, GPUmem=0,70
    26,00%, ll=-0,00820, gf=10,548, secs=0,1, GB=0,02, MB/s=482,65, GPUmem=0,70
    30,00%, ll=-0,00797, gf=10,625, secs=0,1, GB=0,03, MB/s=486,27, GPUmem=0,70
    33,00%, ll=-0,00798, gf=10,685, secs=0,1, GB=0,03, MB/s=489,04, GPUmem=0,70
    36,00%, ll=-0,00795, gf=10,750, secs=0,1, GB=0,03, MB/s=492,29, GPUmem=0,70
    40,00%, ll=-0,00769, gf=10,813, secs=0,1, GB=0,04, MB/s=495,43, GPUmem=0,70
    43,00%, ll=-0,00811, gf=10,718, secs=0,1, GB=0,04, MB/s=491,17, GPUmem=0,70
    46,00%, ll=-0,00824, gf=10,746, secs=0,1, GB=0,04, MB/s=492,30, GPUmem=0,70
    50,00%, ll=-0,00798, gf=10,786, secs=0,1, GB=0,05, MB/s=494,21, GPUmem=0,70
    53,00%, ll=-0,00784, gf=10,802, secs=0,1, GB=0,05, MB/s=494,82, GPUmem=0,70
    56,00%, ll=-0,00809, gf=10,832, secs=0,1, GB=0,05, MB/s=496,25, GPUmem=0,70
    60,00%, ll=-0,00817, gf=9,144, secs=0,1, GB=0,06, MB/s=418,94, GPUmem=0,70
    63,00%, ll=-0,00765, gf=9,239, secs=0,1, GB=0,06, MB/s=423,33, GPUmem=0,70
    66,00%, ll=-0,00818, gf=9,323, secs=0,1, GB=0,06, MB/s=427,19, GPUmem=0,70
    70,00%, ll=-0,00779, gf=9,346, secs=0,2, GB=0,07, MB/s=428,33, GPUmem=0,70
    73,00%, ll=-0,00782, gf=9,418, secs=0,2, GB=0,07, MB/s=431,64, GPUmem=0,70
    76,00%, ll=-0,00761, gf=9,494, secs=0,2, GB=0,07, MB/s=435,24, GPUmem=0,70
    80,00%, ll=-0,00806, gf=9,555, secs=0,2, GB=0,07, MB/s=438,00, GPUmem=0,70
    83,00%, ll=-0,00791, gf=9,559, secs=0,2, GB=0,08, MB/s=438,16, GPUmem=0,70
    86,00%, ll=-0,00812, gf=9,616, secs=0,2, GB=0,08, MB/s=440,77, GPUmem=0,70
    90,00%, ll=-0,00817, gf=9,666, secs=0,2, GB=0,08, MB/s=443,01, GPUmem=0,70
    93,00%, ll=-0,00797, gf=9,711, secs=0,2, GB=0,09, MB/s=445,04, GPUmem=0,70
    96,00%, ll=-0,00817, gf=9,757, secs=0,2, GB=0,09, MB/s=447,12, GPUmem=0,70
    100,00%, ll=-0,00799, gf=9,705, secs=0,2, GB=0,09, MB/s=444,77, GPUmem=0,70
    Time=0,2090 secs, gflops=9,71


The accuracies are :

    0,99035
    0,92883
    0,98513
    0,98612
    0,95681
    0,96348
    ...

To get the training options :

    mopts.what


Command `GPUmem` gives you percentage of used memory, free memory and memory capacity :

    (Float, Long, Long) = (0.69568384,2987802624,4294770688)


Stop the instance :

    aws ec2 terminate-instances --region us-west-2 --instance-ids i-XXX

To get an **updated AMI** with the new version of BIDMach and Cuda 7.5, have a look at my [article about new AMI]({{ site.url }}/big/data/2016/01/27/two-AMI-to-create-the-fastest-cluster-with-gpu-at-the-minimal-engineering-cost-with-EC2-NVIDIA-Spark-and-BIDMach.html).

**Well done!**
