---
layout: post
title:  "2 AMI to launch the fastest cluster of GPU for scientific computing at minimal engineering cost thanks to EC2, Spark, NVIDIA, and BIDMach technologies"
date:   2016-01-27 23:00:51
categories: big data
---

In this tutorial, I will create two AMI for AWS G2 instances (GPU-enabled), the first one with NVIDIA driver and Cuda 7.5 installed, the second one with NVIDIA  driver, Cuda 7.5 and BIDMach technologies.

- **ami-0ef6407d**

- **ami-18f5466b** or with more memory (some files removed): **ami-e2f74491**  


To run an instance from one of these AMI, just run :

{% highlight bash %}
aws ec2 run-instances --image-id ami-e2f74491 --key-name sparkclusterkey \
--instance-type g2.2xlarge --region eu-west-1 --security-groups bidmach
{% endhighlight %}

These AMI are also *Spark-compatible*. In the last section, I'll show how to use them to launch the world's fastest cluster at minimal engineering cost, that is :

- EC2 G2 instances, with NVIDIA GPU offering **1536 cores on a single instance**, the maximal power on a single cloud instance currently available so easily

- [BIDMach library](http://christopher5106.github.io/parallel/computing/2016/01/26/gpu-computing-with-bidmach-library-simply-amazing.html) that unleashes the speed of GPU computing algorithms to **compute on a single instance the equivalent of a cluster** of 8 to 30 instances, depending on the type of machine learning task [[benchmarks](https://github.com/BIDData/BIDMach/wiki/Benchmarks)].

- Spark to launch many of these instances to parallelize the computing along hyperparemeter tuning. Hyperparameter tuning consists in repeating the same machine learning task with different hyperparemeters (parameters for the model). It is a kind of best practice to distribute machine learning computing this way, ie to parallelize along the hyperparemeter tuning (see also [Databricks example for Tensorflow](https://databricks.com/blog/2016/01/25/deep-learning-with-spark-and-tensorflow.html)), each instance will do the training for one set of hyperparameters, instead of distributing the machine learning algorithm itself, which would require a very ineffective data shuffling between instances.

I cannot use AWS EMR to launch the cluster because I need to install nvidia and cuda, that would require a reboot of the instances.

# Creation of the AMI for G2+Spark with NVIDIA driver and CUDA 7.5 installed

Let's launch an instance with AMI Id `ami-2ae0165d` given by [the AMI list for Spark](https://github.com/amplab/spark-ec2/blob/branch-1.5/ami-list/eu-west-1/hvm), for Europe :

{% highlight bash %}
aws ec2 run-instances --image-id ami-2ae0165d --instance-type g2.2xlarge \
--key-name bidmach-keypair --security-groups bidmach
{% endhighlight %}

and connect

{% highlight bash %}
ssh -i bidmach-keypair.pem ec2-user@ec2-XXX.eu-west-1.compute.amazonaws.com
{% endhighlight %}

And install :

{% highlight bash %}
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
{% endhighlight %}

Add /usr/local/cuda/bin to the path in the bash profile to have the NVCC compiler available.

Let's create a public AMI : **ami-0ef6407d**. This AMI can be used

- to test any GPU-capable library on the cloud, such as Theano, Caffe, BIDMach, Tensorflow ...

- to launch with Spark a cluster of GPU instances with NVIDIA Driver and CUDA 7.5 pre-installed, as we show in the last section.


# Compile JCUDA, BIDMat, BIDMach libraries


{% highlight bash %}

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

{% endhighlight %}

Let's create a public AMI : **ami-e2f74491**. This AMI is useful

- to get an instance with BIDMach pre-installed

- to launch with Spark a cluster of GPU instances with NVIDIA driver, CUDA 7.5, JCUDA, BIDMat, and BIDMach libraries installed, as we show below.


<a name="launch_cluster" />

# Launch of a Spark cluster with one of these custom AMI

Let's first fork `https://github.com/amplab/spark-ec2` and create `https://github.com/christopher5106/spark-ec2` repo where I can change the AMI for the previously created AMI `ami-e2f74491`.

Then create an IAM role named *spark-ec2* to later on be able to give access to resources to the Spark cluster (without having to deal with security credentials on the instances - avoiding dangerous `--copy-aws-credentials` option) and add the permission to attribute this role to the user launching the spark-ec2 command :

{% highlight json %}
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
{% endhighlight %}

Create the cluster with the new repo and the instance profile  :

{% highlight bash %}
./ec2/spark-ec2 -k sparkclusterkey -i ~/sparkclusterkey.pem \
--region=eu-west-1 --instance-type=g2.2xlarge \
-s 1 --hadoop-major-version=2 \
--spark-ec2-git-repo=https://github.com/christopher5106/spark-ec2 \
--instance-profile-name=spark-ec2 \
launch spark-cluster
{% endhighlight %}


And log in and start the shell :

{% highlight bash %}
./ec2/spark-ec2 -k sparkclusterkey -i ~/sparkclusterkey.pem \
--region=eu-west-1 login spark-cluster

./spark/bin/spark-shell
{% endhighlight %}

Terminate the cluster:

{% highlight bash %}
./ec2/spark-ec2 -k sparkclusterkey -i ~/sparkclusterkey.pem \
--region=eu-west-1  destroy spark-cluster
{% endhighlight %}
