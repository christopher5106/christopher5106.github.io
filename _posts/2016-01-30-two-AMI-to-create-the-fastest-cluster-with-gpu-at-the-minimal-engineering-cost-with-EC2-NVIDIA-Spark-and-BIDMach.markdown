---
layout: post
title:  "3 AMI to run the fastest cluster of GPU for scientific computing at minimal engineering cost thanks to EC2, Spark, NVIDIA, and BIDMach technologies"
date:   2016-01-27 23:00:51
categories: big data
---

In this tutorial, I will create three AMI for AWS G2 instances (GPU-enabled), the first one for any use of NVIDIA technologies, the second one for the use of [scala fast GPU library BIDMach](http://christopher5106.github.io/big/data/2016/02/04/bidmach-tutorial.html) and the third one for the creation of a cluster of GPU instances with BIDMach:

- **ami-0ef6407d** : NVIDIA driver + Cuda 7.5
This first AMI is *Spark-ec2 compatible*, it can be launched with *spark-ec2* command.

- **ami-18f5466b** (or **ami-e2f74491** for a memory efficient AMI - some unnecessary files removed): NVIDIA driver + Cuda 7.5 + BIDMach
Since Spark-ec2 installs Spark for Scala 2.10, this AMI cannot be used with *spark-ec2* command.

- **ami-dbfa48a8** :  NVIDIA driver + Cuda 7.5 + BIDMach + Spark
This AMI enables to create a cluster of GPU instances with BIDMach.

To run an instance from one of these AMI, just run :


```bash
aws ec2 run-instances --image-id ami-e2f74491 --key-name sparkclusterkey \
--instance-type g2.2xlarge --region eu-west-1 --security-groups bidmach
```

Placing many of these instances in a cluster make the fastest cluster at minimal engineering cost, that is :

- EC2 G2 instances, with NVIDIA GPU offering **1536 cores on a single instance**, the maximal power on a single cloud instance currently available so easily

- [BIDMach library](http://christopher5106.github.io/parallel/computing/2016/01/26/gpu-computing-with-bidmach-library-simply-amazing.html) that unleashes the speed of GPU computing algorithms to **compute on a single instance at the speed of a cluster** of 8 to 30 instances, depending on the type of machine learning task [[benchmarks](https://github.com/BIDData/BIDMach/wiki/Benchmarks)].

- Spark to launch many of these instances and **parallelize the computing along hyperparemeter tuning**. Hyperparameter tuning consists in repeating the exact same machine learning task but with different parameters for the model (the *hyperparemeters*). It is a kind of best practice to distribute machine learning computing along hyperparemeter tuning (each instance does the training for a set of hyperparameters), instead of distributing the training task itself (see also [Databricks example for Tensorflow](https://databricks.com/blog/2016/01/25/deep-learning-with-spark-and-tensorflow.html)), because it splits the computation in jobs (training) that do not need to communicate and avoids this way the ineffective data shuffling between instances.

I cannot use AWS EMR to launch a GPU cluster because I need to install nvidia and cuda, that would require a reboot of the instances.


# Creation of the AMI for G2+Spark with NVIDIA driver and CUDA 7.5 installed

Let's launch an instance with AMI Id `ami-2ae0165d` given by [the AMI list for Spark](https://github.com/amplab/spark-ec2/blob/branch-1.5/ami-list/eu-west-1/hvm), for Europe :

```bash
aws ec2 run-instances --image-id ami-2ae0165d --instance-type g2.2xlarge \
--key-name bidmach-keypair --security-groups bidmach
```

and connect

```bash
ssh -i bidmach-keypair.pem ec2-user@ec2-XXX.eu-west-1.compute.amazonaws.com
```

And install NVIDIA driver and CUDA 7.5 :

```bash
sudo yum update -y
sudo reboot
sudo yum groupinstall -y "Development tools"
sudo yum install kernel-devel-`uname -r`

# install the driver
wget http://us.download.nvidia.com/XFree86/Linux-x86_64/352.79/NVIDIA-Linux-x86_64-352.79.run
sudo /bin/bash NVIDIA-Linux-x86_64-352.79.run
sudo reboot
# check if it works
nvidia-smi -q | head

# install Cuda
sudo mkdir /mnt/tmp
wget http://developer.download.nvidia.com/compute/cuda/7.5/Prod/local_installers/cuda_7.5.18_linux.run
sudo bash cuda_7.5.18_linux.run --tmpdir /mnt/tmp
```

Add */usr/local/cuda/bin* to the PATH variable in the bash profile to have the NVCC compiler available.

Let's create a public AMI : **ami-0ef6407d**. This AMI can be used

- to use any GPU-capable library on the cloud, such as Theano, Caffe, BIDMach, Tensorflow ...

- to launch with Spark a cluster of GPU instances with NVIDIA Driver and CUDA 7.5 pre-installed, as we show now:

<a name="launch_cluster" />

To launch a cluster of this AMI :

-  first fork `https://github.com/amplab/spark-ec2` and in the newly created `https://github.com/christopher5106/spark-ec2` repo, I change the AMI for the previously created AMI `ami-e2f74491`

- then create an IAM role named *spark-ec2* to later on be able to give access to resources to the Spark cluster (without having to deal with security credentials on the instances - avoiding dangerous `--copy-aws-credentials` option) and add the permission to attribute this role to the user launching the spark-ec2 command :

```json
{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect":"Allow",
      "Action":["ec2:*"],
      "Resource":"*"
    },
        {
            "Effect": "Allow",
            "Action": "iam:PassRole",
            "Resource": "arn:aws:iam::ACCOUNT_ID:role/spark-ec2"
        }
    ]
}
```

- create the cluster with the new repo and the instance profile  :

```bash
./ec2/spark-ec2 -k sparkclusterkey -i ~/sparkclusterkey.pem \
--region=eu-west-1 --instance-type=g2.2xlarge \
-s 1 --hadoop-major-version=2 \
--spark-ec2-git-repo=https://github.com/christopher5106/spark-ec2 \
--instance-profile-name=spark-ec2 \
launch spark-cluster
```


And log in and start the shell :

```bash
./ec2/spark-ec2 -k sparkclusterkey -i ~/sparkclusterkey.pem \
--region=eu-west-1 login spark-cluster

./spark/bin/spark-shell
```

Terminate the cluster:

```bash
./ec2/spark-ec2 -k sparkclusterkey -i ~/sparkclusterkey.pem \
--region=eu-west-1  destroy spark-cluster
```


# Creation of the AMI for G2 with NVIDIA driver, CUDA 7.5, JCUDA, BIDMat and BIDMach libraries pre installed

Compile JCUDA, BIDMat, BIDMach libraries on the previous instance :

```bash
#install Cmake
wget https://cmake.org/files/v3.4/cmake-3.4.3.tar.gz
tar xvzf cmake-3.4.3.tar.gz
cd cmake-3.4.3
./configure
make
sudo make install
cd ..

#install maven
wget http://apache.mindstudios.com/maven/maven-3/3.3.9/binaries/apache-maven-3.3.9-bin.tar.gz
tar xvzf apache-maven-3.3.9-bin.tar.gz
export PATH=$PATH:/home/ec2-user/apache-maven-3.3.9/bin #not a best practice!

# install Intel Parallel Studio
wget LLLLL/parallel_studio_xe_2016_composer_edition_for_cpp_update1.tgz
tar xvzf parallel_studio_xe_2016_composer_edition_for_cpp_update1.tgz

#compile JCuda
mkdir JCuda
cd JCuda
git clone https://github.com/jcuda/jcuda-common.git
git clone https://github.com/jcuda/jcuda-main.git
git clone https://github.com/jcuda/jcuda.git
git clone https://github.com/jcuda/jcublas.git
git clone https://github.com/jcuda/jcufft.git
git clone https://github.com/jcuda/jcusparse.git
git clone https://github.com/jcuda/jcurand.git
git clone https://github.com/jcuda/jcusolver.git
cmake jcuda-main
sudo yum install mesa-libGL-devel
make
cd jcuda-main/
mvn clean install
cd ../..

#compile BIDMat
git clone https://github.com/BIDData/BIDMat.git
cd BIDMat
./getdevlibs.sh
cd jni/src/
./configure
make
make install
cd ../..
./sbt clean package
cd ..

#compite BIDMach
git clone https://github.com/BIDData/BIDMach.git
cd BIDMach
./getdevlibs.sh
cp ../BIDMat/BIDMat.jar lib/
cp ../JCuda/jcuda/JCudaJava/target/lib/libJCuda* lib/
cd jni/src
./configure
make
make install
cd ../..
./sbt clean package

```

Let's create a public AMI : **ami-18f5466b**. This AMI is useful to get an instance with BIDMach pre-installed.

Be careful :

- before launching `bidmach` command, change the version and augment the memory in the *bidmach* file :

    ```bash
    JCUDA_VERSION="0.7.5" # Fix if needed
    MEMSIZE="-Xmx12G"
    ```

- if you intend to use Spark, you need to compile BIDMach for Scala 2.10 (which is not supported), otherwise see in the next section (I compile Spark for Scala 2.11 and create a third AMI).

Let's delete BIDMat, BIDMach tutorials, Maven, JCuda, to create a more efficient AMI with more space **ami-e2f74491**, which is definitely the one you should choose.


<a name="cluster_of_gpu" />

# Creation of the AMI for a cluster of GPU G2 with Spark and NVIDIA driver, CUDA 7.5, JCUDA, BIDMat, BIDMach libraries pre installed

To have Spark work with BIDMach, I compile Spark with Scala 2.11, since BIDMach is only supported for Scala 2.11 :

```bash
wget http://apache.crihan.fr/dist/spark/spark-1.6.0/spark-1.6.0.tgz
tar xvzf spark-1.6.0.tgz
cd spark-1.6.0
./dev/change-scala-version.sh 2.11
export MAVEN_OPTS='-Xmx2g -XX:MaxPermSize=2g'
mvn -Pyarn -Phadoop-2.6 -Dscala-2.11 -DskipTests clean package
export SPARK_HOME=`pwd`
```

Let's create the AMI **ami-dbfa48a8** and launch two instances of them.

Let's connect to the first (the master) and create the *conf/slaves* file with one line corresponding to the public DNS of our second :

    ec2-54-229-106-189.eu-west-1.compute.amazonaws.com

On both of them, let's create a directory */mnt/spark* with *ec2-user* as owner, and define *conf/spark-env.sh* file :

```
#!/usr/bin/env bash

export SPARK_LOCAL_DIRS="/mnt/spark"

# Standalone cluster options
export SPARK_MASTER_OPTS=""
if [ -n "1" ]; then
  export SPARK_WORKER_INSTANCES=1
fi
export SPARK_WORKER_CORES=2

export SPARK_MASTER_IP=ec2-54-229-155-126.eu-west-1.compute.amazonaws.com
export MASTER=ec2-54-229-155-126.eu-west-1.compute.amazonaws.com

# Bind Spark's web UIs to this machine's public EC2 hostname otherwise fallback to private IP:
export SPARK_PUBLIC_DNS=`
wget -q -O - http://169.254.169.254/latest/meta-data/public-hostname ||\
wget -q -O - http://169.254.169.254/latest/meta-data/local-ipv4`

# Set a high ulimit for large shuffles, only root can do this
if [ $(id -u) == "0" ]
then
    ulimit -n 1000000
fi
```

On the master, launch the cluster with `sbin/start-all.sh`.

Set your master and slave security groups on the instances.

Now it's time to launch a shell :

```bash
sudo ./bin/spark-shell \
--master=spark://ec2-54-229-155-126.eu-west-1.compute.amazonaws.com:7077 \
--jars /home/ec2-user/BIDMach/BIDMach.jar,/home/ec2-user/BIDMach/lib/BIDMat.jar,/home/ec2-user/BIDMach/lib/jhdf5.jar,/home/ec2-user/BIDMach/lib/commons-math3-3.2.jar,/home/ec2-user/BIDMach/lib/lz4-1.3.jar,/home/ec2-user/BIDMach/lib/json-io-4.1.6.jar,/home/ec2-user/BIDMach/lib/jcommon-1.0.23.jar,/home/ec2-user/BIDMach/lib/jcuda-0.7.5.jar,/home/ec2-user/BIDMach/lib/jcublas-0.7.5.jar,/home/ec2-user/BIDMach/lib/jcufft-0.7.5.jar,/home/ec2-user/BIDMach/lib/jcurand-0.7.5.jar,/home/ec2-user/BIDMach/lib/jcusparse-0.7.5.jar \
--driver-library-path="/home/ec2-user/BIDMach/lib" \
--conf "spark.executor.extraLibraryPath=/home/ec2-user/BIDMach/lib"
```

**Our cluster of GPU is ready!**. Go on with a [Random forest computation on the cluster](http://christopher5106.github.io/big/data/2016/02/04/bidmach-tutorial.html).
