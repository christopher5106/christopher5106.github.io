---
layout: post
title:  "Big data tutorial on BIDMach library : basic matrix operations and file I/O. Example on a RandomForest computation"
date:   2016-02-04 23:00:51
categories: big data
---

BIDMach is a very powerful library for Scala,

- on matrix operations, as powerful as Python Numpy library on CPU, but offering to possibility to work on CPU and GPU indifferently.

- on file i/o, in particular with [lz4 compression](https://github.com/Cyan4973/lz4).

Let's put things back in order :)

First, we'll see basic matrices, then more complicated types, then how to prepare data and an example of random forest computation.

To launch BIDMach on your computer or instance, have a look [at my previous article]({{ site.url }}/parallel/computing/2016/01/26/gpu-computing-with-bidmach-library-simply-amazing.html).

# Basic matrices

Let's create our first matrix of integers of size 2x2, then a matrix of floats of size 3x2, a matrix of double values of size 2x2 and a matrix of strings :

{% highlight scala %}
val imat = IMat(2,2, Array(1,2,3,4))
val fmat = FMat(2,3, Array(1,2,3,4,5,6))
val dmat = DMat(2,2)
val smat = CSMat(1,3, Array("you","and","me"))
{% endhighlight %}

Access the number of columns, rows :

{% highlight scala %}
size(fmat)
fmat.dims
fmat.length
fmat.ncols
fmat.nrows
{% endhighlight %}

Two ways to access element on (row 1, column 2), either by *row x column* tuple, or by *index* (column-oriented numbering) :

{% highlight scala %}
fmat(0,1)
fmat(2)
{% endhighlight %}

Access all elements as one column :

{% highlight scala %}
fmat(?)
{% endhighlight %}

Access first column and first row :

{% highlight scala %}
fmat(?,0)
fmat(0,?)
{% endhighlight %}

Access elements with indexes between 1 and 3 (not inclusive):

{% highlight scala %}
fmat(1->3)
{% endhighlight %}

Create a Matrix of size 2x2, with 3rd element on first position, 4th on second, 2nd on third position and 5th on last position :

{% highlight scala %}
fmat(IMat(2,2,Array( 2,3,1,5 ) ))
{% endhighlight %}

Create a random, full-one and full-zero matrices of shape 2x2 :

{% highlight scala %}
rand(2,2)
ones(2,2)
zeros(2,2)
{% endhighlight %}

As a shortcut to `IMat(len, 1, Array(values))` to create a single-column matrix :

{% highlight scala %}
icol(1,2,3,4,5)
1 on 2 on 3 on 4 on 5
{% endhighlight %}

The same for single-row matrices :

{% highlight scala %}
irow(1,2,3,4)
1 \ 2 \ 3 \ 4
{% endhighlight %}

As with Mat, you'll find `col`, `dcol`, `cscol`, and `row`, `drow`, `csrow` types.

To get the diagonal : `getdiag(fmat)`

To make a square matrix with a given diagonal : `mkdiag(1 on 2)`

Element-wise matrix operations :

{% highlight scala %}
a + b // element-wise addition
a - b // element-wise subtraction
b ∘ a // (a *@ b) multiplication
b / a /* element wise-division */
{% endhighlight %}

Matrix operation

{% highlight scala %}
a.t // transpose
a * b // matrix multiplication
a ^* b  // transpose first matrix before multiplication
a *^ b  /* transpose second matrix before multiplication */
{% endhighlight %}

Dot products

{% highlight scala %}
a ∙ b // (a dot b) Column-wise dot product
a ∙→ b /* (a dotr b) Row-wise dot product */
{% endhighlight %}

Cartesian product

{% highlight scala %}
a ⊗ b /* (a kron b) Kronecker product */
{% endhighlight %}

Statistics per column :

{% highlight scala %}
sum(fmat)
mean(fmat)
variance(fmat)
maxi(fmat)
maxi2(fmat) // returns max and argmax
mini(fmat)
mini2(fmat) /* returns min and argmin */
{% endhighlight %}

Other operations per column :

{% highlight scala %}
cumsum(fmat)
sort(fmat) // sorted values
sortdown(fmat) // sorted values
sort2(fmat) // sorted values and indices
sortdown2(fmat) /* sorted values and indices */
{% endhighlight %}

The same statistics and operations per rows by adding a 2 param :

{% highlight scala %}
sum(fmat,2)
{% endhighlight %}

Have a look at unique(), unique3() and uniquerows().

Endly, to reshape, you can convert it to a n-dimension array, reshape it, and convert the result array back to a FMat :

{% highlight scala %}
FND(fmat).reshape(3,2).toFMat(3,2)
{% endhighlight %}


# More complicated matrices

Convert to a GPU matrix :

{% highlight scala %}
GMat(fmat)
GIMat(imat)
{% endhighlight %}

Create a generic type matrix :

{% highlight scala %}
var g:Mat = null
g = fmat
g = GMat(fmat)
{% endhighlight %}

Convert to sparse format :

{% highlight scala %}
val sfmat = SMat(fmat)
val sfmat = sparse(fmat)
SDmat(fmat) // sparse double
GSMat(fmat) /* sparse GPU */
{% endhighlight %}

To convert it back to a dense matrix :

{% highlight scala %}
full(sfmat)
{% endhighlight %}

Create a sparse matrix with value 5 at position (2,4) and value 6 at position (3,4) :

{% highlight scala %}
sparse(1\2,3\3, 5\6 ,  4,4)
{% endhighlight %}

Create an accumulator matrix of shape (3,3), by accumulating 5 and 4 on the same cell (3,2):

{% highlight scala %}
accum( (2\1) on (2\1), 5 on 4, 3,3)
{% endhighlight %}

Enable matrix caching :

{% highlight scala %}
Mat.useCache = true
{% endhighlight %}

There is also a great tool, the dictionaries, where counts is an optional argument :

{% highlight scala %}
val d = Dict(CSMat(1,2,Array("A","B")),IMat(1,2, Array(5,4)))
d(0) // value for index 0 : A
d("A") // index of A : 0
d.count("A") // counts : 5
d.count(0) // same
d.cstr // dictionary
d.counts // counts
d(IMat(1,2,Array(0,1))) // bulk retrieval
d(CSMat(1,2,Array("A","B)))
{% endhighlight %}


# File I/O


Save the matrix `fmat` as a text file :

{% highlight scala %}
saveFMat("t2.txt",fmat,1)
{% endhighlight %}

With gz compression :

{% highlight scala %}
saveFMat("t.gz",fmat)
{% endhighlight %}

With lz4 compression :

{% highlight scala %}
saveFMat("t.lz4",fmat)
{% endhighlight %}

To load the file :

{% highlight scala %}
loadFMat("t.lz4")
{% endhighlight %}

Have a look at load (and saveAs), loadLibSVM, loadIDX for HDF5, LibSVM and IDX format.

For efficient I/O, convert the string matrices to sparce matrices of bytes with :

{% highlight scala %}
SBMat(smat)
{% endhighlight %}

# Others

BIDMach offers also

- many random functions : randperm(), rand(), normrnd(),...

- a timer and flops counter : `tic` to start/reset timer, `flip` to start/reset timer and gflop counter, `toc` to get time since, `flop` to get time and flops counter, `gflop` for GPU

- cache management

- complex types

- plot possibilities :

{% highlight scala %}
plot( 1\2\3\4\5,3.2\2.1\1.3\4.2\(-1.2))
{% endhighlight %}

![]({{ site.url }}/img/bidmach_plot.png)

{% highlight scala %}
show(Image(IMat(2,2,Array(255,0,0,255)   ) kron ones(100,100) ))
{% endhighlight %}

![]({{ site.url }}/img/bidmach_image_show.png)

# Data sources

A data source is an iterator, let's create 5 data with their respective label (0), and pull the data iteratively 2 by 2 :

{% highlight scala %}
val dataimat = irow(1 to 5)
val labelimat = izeros(1,5)

val dopts = new MatSource.Options
dopts.batchSize = 2
val m = new MatSource(Array(dataimat, labelimat),dopts)
m.init
m.hasNext
val value = m.next
{% endhighlight %}

The output `m.next` is an array of 2 IMat, the first one is the data, the second the label. `dopts.what` will give you all options.

Option `dopts.sample` enables to sample of fraction of the data, and `opts.addConstFeat` adds a feature with constant value to 1 to the data for regression bias.

Data sources can also be used as a sink, to put back some data, such as the prediction value

{% highlight scala %}
dopts.putBack = 1  
val predimat = izeros(1,5)
val m = new MatSource(Array(dataimat, predimat),dopts)
m.init
val value = m.next
value(1)(1)=1
m.next
m.mats
{% endhighlight %}

Let's now go further with file data sources.

For the purpose, let's create 50 files of data, 50 files of labels, with random values for the explicative variables, and a weighted sum for the label, for the demonstration :

{% highlight scala %}
val params = drand(30,1)

0 until 50 map( i => {
  val fmat = FMat(drand(30,1000))
  saveMat("data%02d.fmat.lz4" format i, fmat);
  saveMat("label%02d.imat.lz4" format i, IMat(params ^* fmat) > 7)
})
{% endhighlight %}

Note the use of `until`, while `to` would create 51 values.

Check matrices are correctly saved with : `loadMat("data26.fmat.lz4")`. Now there are correctly saved, let's create a file data source :

{% highlight scala %}
val fopts = new FileSource.Options
fopts.fnames = List( {i:Int => {"data%02d.fmat.lz4" format i}}, {i:Int => {"label%02d.imat.lz4" format i}} )
fopts.nstart = 0
fopts.nend = 50
fopts.batchSize = 1000

val fs = FileSource(fopts)

fs.init
fs.hasNext
val value = fs.next
{% endhighlight %}


`fopts.order=1` randomize the order of the columns.

`fopts.what` give all set options.

Lastly, have a look at SFileSource for sparse file data source.


# Run a Random Forest regressor


On the files created previously, let's launch the random forest regressor :

{% highlight scala %}
val (mm,opts) = RandomForest.learner("data%02d.fmat.lz4","label%02d.imat.lz4")

opts.batchSize = 1000
opts.nend = 50
opts.depth =  5
opts.ncats = 2 // number of categories of label
opts.ntrees = 20
opts.impurity = 0
opts.nsamps = 12
opts.nnodes = 50000
opts.nbits = 16
opts.gain = 0.001f
mm.train

{% endhighlight %}

Option `opts.useGPU = false` will disable use of GPU.

    pass= 0
    40,00%, ll=-0,02400, gf=0,250, secs=0,0, GB=0,00, MB/s= 9,92, GPUmem=0,971549
    100,00%, ll=-0,02400, gf=0,407, secs=0,0, GB=0,00, MB/s=13,48, GPUmem=0,971549
    purity gain 0,0632, fraction impure 1,000, nnew 2,0, nnodes 3,0
    pass= 1
    40,00%, ll=-0,02400, gf=0,302, secs=0,1, GB=0,00, MB/s=10,46, GPUmem=0,971549
    100,00%, ll=-0,02400, gf=0,369, secs=0,1, GB=0,00, MB/s=12,16, GPUmem=0,971549
    purity gain 0,0425, fraction impure 1,000, nnew 2,2, nnodes 5,2
    pass= 2
    40,00%, ll=-0,02400, gf=0,321, secs=0,1, GB=0,00, MB/s=10,86, GPUmem=0,971549
    100,00%, ll=-0,02400, gf=0,367, secs=0,2, GB=0,00, MB/s=12,08, GPUmem=0,971549
    purity gain 0,1401, fraction impure 0,834, nnew 4,1, nnodes 9,3
    pass= 3
    40,00%, ll=-0,00700, gf=0,334, secs=0,2, GB=0,00, MB/s=11,15, GPUmem=0,971549
    100,00%, ll=-0,00700, gf=0,364, secs=0,2, GB=0,00, MB/s=11,92, GPUmem=0,971549
    purity gain 0,1843, fraction impure 0,807, nnew 5,5, nnodes 14,8
    pass= 4
    40,00%, ll=-0,00300, gf=0,341, secs=0,2, GB=0,00, MB/s=11,32, GPUmem=0,971549
    100,00%, ll=-0,00300, gf=0,365, secs=0,3, GB=0,00, MB/s=11,92, GPUmem=0,971549
    purity gain 0,1463, fraction impure 0,599, nnew 4,9, nnodes 19,7
    Time=0,2820 secs, gflops=0,34


Parameters are :

- depth(20): Bound on the tree depth, also the number of passes over the dataset.

- ntrees(20): Number of trees in the Forest.

- nsamps(32): Number of random features to try to split each node.

- nnodes(200000): Bound on the size of each tree (number of nodes).

- nbits(16): Number of bits to use for feature values.

- gain(0.01f): Lower bound on impurity gain in order to split a node.

- catsPerSample(1f): Number of cats per sample for multilabel classification.

- ncats(0): Number of cats or regression values. 0 means guess from datasource.

- training(true): Run for training (true) or prediction (false)

- impurity(0): Impurity type, 0=entropy, 1=Gini

- regression(false): Build a regression Forest (true) or classification Forest (false).

- seed(1): Random seed for selecting features. Use this to train distinct Forests in multiple runs.

- useIfeats(false): An internal var, when true use explicit feature indices vs compute them.

- MAE(true): true=Use Mean Absolute Error when reporting performance vs. false=Mean Squared Error

- trace(0): level of debugging information to print (0,1,2).

<a name="spark" />


# Run BIDMach on Spark

Instead of running `bidmach` command to launch a bidmach shell, it is possible to run the same commands inside Spark shell, adding the BIDMat and BIDMach libraries to the classpath.

To have Spark work with BIDMach, compile Spark with Scala 2.11 :

{% highlight bash %}
wget http://apache.crihan.fr/dist/spark/spark-1.6.0/spark-1.6.0.tgz
tar xvzf spark-1.6.0.tgz
cd spark-1.6.0
./dev/change-scala-version.sh 2.11
mvn -Pyarn -Phadoop-2.6 -Dscala-2.11 -DskipTests clean package
export SPARK_HOME=`pwd`
{% endhighlight %}

Download [Joda-time](https://sourceforge.net/projects/joda-time/files/joda-time/) and launch Spark in **local mode** with the Amazon SDK, Joda-time, and BIDMat / BIDMach jars :

{% highlight bash %}
$SPARK_HOME/bin/spark-shell --jars \
~/technologies/aws-java-sdk-1.10.51/lib/aws-java-sdk-1.10.51.jar,\
../../technologies/BIDMach2/lib/BIDMat.jar,\
../../technologies/BIDMach2/BIDMach.jar,\
../../technologies/joda-time-2.4/joda-time-2.4.jar
{% endhighlight %}

Then you can import the required libraries :

{% highlight scala %}
import BIDMach.models.RandomForest
{% endhighlight %}

You can also launch a **cluster of GPU**, for example 2 g2.2xlarge executor instances with our [NVIDIA+CUDA+BIDMACH AMI for Spark](http://christopher5106.github.io/big/data/2016/01/27/two-AMI-to-create-the-fastest-cluster-with-gpu-at-the-minimal-engineering-cost-with-EC2-NVIDIA-Spark-and-BIDMach.html):

{% highlight bash %}
$SPARK_HOME/ec2/spark-ec2 -k sparkclusterkey -i ~/sparkclusterkey.pem \
--region=eu-west-1 \
 --instance-type=g2.2xlarge -s 2 \
--hadoop-major-version=2  \
--spark-ec2-git-repo=https://github.com/christopher5106/spark-ec2 \
--instance-profile-name=spark-ec2 \
launch spark-cluster
{% endhighlight %}

Note that I launch the instances under an IAM role named *spark-ec2* to give them access to resources later on (without having to deal with security credentials on the instance - avoid using `--copy-aws-credentials` option), as [explained in my previous post]({{ site.url }}/big/data/2016/01/27/two-AMI-to-create-the-fastest-cluster-with-gpu-at-the-minimal-engineering-cost-with-EC2-NVIDIA-Spark-and-BIDMach.html#launch_cluster).

And log in

{% highlight bash %}
./ec2/spark-ec2 -k sparkclusterkey -i ~/sparkclusterkey.pem \
--region=eu-west-1 \
login spark-cluster
{% endhighlight %}

Launch the Spark Shell and be sure to have only 1 core per GPU on each executor :

{% highlight bash %}
./spark/bin/spark-shell --conf spark.executor.cores=1 --jars \
 /home/ec2-user/BIDMach/lib/BIDMat.jar,/home/ec2-user/BIDMach/BIDMach.jar
{% endhighlight %}


<a name="spark_prepare_data" />

# Prepare the data with Spark

Spark is ideal to split very large data files into smaller parts that will be saved in BIDMach file format to feed the BIDMach file data sources.

Spark naturally splits the input file into parts for each job, that are accessible via **mapPartitions** methods, in order to execute a custom function on each split, as for example :

{% highlight scala %}
val file = sc.textFile("myfile.csv")
val header_line = file.first()
val tail_file = file.filter( _ != header_line)
val allData = tail_file.mapPartitionsWithIndex( upload_lz4_fmat_to_S3 )
allData.collect()
{% endhighlight %}

Adjusting the parallelism (number of partitions) will adjust the split size / number of lz4 files.

To build the custom *upload_lz4_fmat_to_S3* method, let's first create a function to upload to S3 :

{% highlight scala %}
def upload_file_to_s3(filepath:String, bucket:String, directory:String) : Int = {
  import java.io.File
  import com.amazonaws.services.s3.AmazonS3Client
  import com.amazonaws.services.s3.model.PutObjectRequest;
  import com.amazonaws.AmazonClientException;
  import com.amazonaws.AmazonServiceException;

  val S3Client = new AmazonS3Client()
  val fileToUpload = new File(filepath)

  try {
    S3Client.putObject(new PutObjectRequest(bucket, directory + "/" + filepath.split("/").last, fileToUpload))
  } catch {
    case ex: AmazonServiceException =>{
      println("Amazon Service Exception : "+ex.getMessage())
      return -1
    }
    case ex: AmazonClientException => {
      println("Amazon Client Exception + " + ex.getMessage())
      return -1
    }
  }
  return 0
}
{% endhighlight %}

The Amazon SDK looks for the credentials available in the different contexts.

Then, define a function to convert a line (String) of data from a CSV into an array of explicative features and the label :

{% highlight scala %}
def convert_line_to_Expl_Label_Tuple(line : String) : (Array[Float],Float) = {
  val values = line.split(";")

  // process your line HERE

  (expl, label)
}
{% endhighlight %}

Lastly, combine the functions to create  *upload_lz4_fmat_to_S3* method :

{% highlight scala %}
def upload_lz4_fmat_to_S3 (index:Int, it:Iterator[String]) : Iterator[Int] = {
  import BIDMat.FMat
  import BIDMat.MatFunctions._
  val dataWithLabel = it.toArray.map( convert_line_to_Expl_Label_Tuple )
  val data = dataWithLabel.flatMap( x => x._1)
  val labels = dataWithLabel.map( x => x._2 )
  val datafmat = FMat(nb_expl, data.length/nb_expl, data)
  val labelfmat = FMat(1,labels.length, labels)
  saveFMat("data%02d.lz4" format index, datafmat)
  saveFMat("label%02d.lz4" format index, labelfmat)
  Array( upload_file_to_s3("data%02d.lz4" format index,bucket, "out"),upload_file_to_s3("label%02d.lz4" format index,bucket, "out") ).iterator
}
{% endhighlight %}


# Hyperparameter tuning using grid search with Spark

Hyperparameters are the parameters of the prediction model : number of trees, depth, number of bins... for example for a random forest regressor.

Grid search consists in computing the model for a grid of combinations for the parameters, for example

{% highlight scala %}
val lrates = col(0.03f, 0.1f, 0.3f, 1f)        // 4 values
val texps = col(0.3f, 0.4f, 0.5f, 0.6f, 0.7f)  // 5 values

val lrateparams = ones(texps.nrows, 1) ⊗ lrates
val texpparams = texps ⊗ ones(lrates.nrows,1)
lrateparams \ texpparams
{% endhighlight %}
