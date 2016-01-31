---
layout: post
title:  "2 AMI to launch the fastest cluster of GPU for scientific computing at minimal engineering cost thanks to EC2, Spark, NVIDIA, and BIDMach technologies"
date:   2016-01-27 23:00:51
categories: big data
---

To create the world's fastest cluster at minimal engineering cost, let's use:

- EC2 G2 instances, with NVIDIA GPU offering **1536 cores on a single instance**, the maximal power on a single cloud instance currently available so easily

- [BIDMach library](http://christopher5106.github.io/parallel/computing/2016/01/26/gpu-computing-with-bidmach-library-simply-amazing.html) that unleashes the speed of GPU computing algorithms to **compute on a single instance the equivalent of a cluster** of 8 to 30 instances, depending on the type of machine learning task [[benchmarks](https://github.com/BIDData/BIDMach/wiki/Benchmarks)].

- Spark to launch many of these instances to parallelize the computing along hyperparemeter tuning. Hyperparameter tuning consists in repeating the same machine learning task with different hyperparemeters (parameters for the model). It is a kind of best practice to distribute machine learning computing this way, ie to parallelize along the hyperparemeter tuning (see also [Databricks example for Tensorflow](https://databricks.com/blog/2016/01/25/deep-learning-with-spark-and-tensorflow.html)), each instance will do the training for one set of hyperparameters, instead of distributing the machine learning algorithm itself, which would require a very ineffective data shuffling between instances.

Sadly, I cannot use AWS EMR to launch the cluster because I need to install nvidia and cuda, that would require a reboot of the instances.



# Create an AMI for Spark with NVIDIA driver and CUDA 7.5 installed

Let's launch an instance with AMI Id `ami-2ae0165d` given by [the AMI list for Spark](https://github.com/amplab/spark-ec2/blob/branch-1.5/ami-list/eu-west-1/hvm), for Europe :

    aws ec2 run-instances --image-id ami-2ae0165d --instance-type g2.2xlarge  --key-name bidmach-keypair --security-groups bidmach

and connect

    ssh -i bidmach-keypair.pem ec2-user@ec2-54-229-72-6.eu-west-1.compute.amazonaws.com

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

Let's create a public AMI : **ami-0ef6407d**.

Now you can use this AMI to launch a cluster of G2 instances.

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



# Launch the cluster

Let's fork `https://github.com/amplab/spark-ec2` and create `https://github.com/christopher5106/spark-ec2` repo where I can change the AMI for the previously created AMI `ami-0ef6407d` and specify this repo to Spark for the creation of the cluster :

    ./ec2/spark-ec2 -k sparkclusterkey -i ~/sparkclusterkey.pem --region=eu-west-1 --copy-aws-credentials --instance-type=g2.2xlarge -s 1 --hadoop-major-version=2 --spark-ec2-git-repo=https://github.com/christopher5106/spark-ec2 launch spark-cluster


Terminate :

    ./ec2/spark-ec2 -k sparkclusterkey -i ~/sparkclusterkey.pem --region=eu-west-1  destroy spark-cluster
