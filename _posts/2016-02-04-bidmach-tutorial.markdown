---
layout: post
title:  "Big data tutorial on BIDMach library : basic matrix operations and file I/O. Example on a RandomForest computation in a cluster of GPU"
date:   2016-02-04 23:00:51
categories: big data
---

BIDMach is a very powerful computation library in Scala,

- on matrix operations, as powerful as Python Numpy library on CPU, but offering to possibility to work on CPU and GPU indifferently.

- on file i/o, in particular with [lz4 compression](https://github.com/Cyan4973/lz4).

Let's put things back in order :)

First, we'll see basic matrices, then more complicated types, then how to prepare data and an example of random forest computation.

To launch BIDMach on your computer or instance, have a look [at my previous article]({{ site.url }}/parallel/computing/2016/01/26/gpu-computing-with-bidmach-library-simply-amazing.html).

# Basic matrices

Let's create our first matrix of integers of size 2x2, then a matrix of floats of size 3x2, a matrix of double values of size 2x2 and a matrix of strings :

```scala
val imat = IMat(2,2, Array(1,2,3,4))
val fmat = FMat(2,3, Array(1,2,3,4,5,6))
val dmat = DMat(2,2)
val smat = CSMat(1,3, Array("you","and","me"))
```

Access the number of columns, rows :

```scala
size(fmat)
fmat.dims
fmat.length
fmat.ncols
fmat.nrows
```

Two ways to access element on (row 1, column 2), either by *row x column* tuple, or by *index* (column-oriented numbering) :

```scala
fmat(0,1)
fmat(2)
```

Access all elements as one column :

```scala
fmat(?)
```

Access first column and first row :

```scala
fmat(?,0)
fmat(0,?)
```

Access elements with indexes between 1 and 3 (not inclusive):

```scala
fmat(1->3)
```

Create a Matrix of size 2x2, with 3rd element on first position, 4th on second, 2nd on third position and 5th on last position :

```scala
fmat(IMat(2,2,Array( 2,3,1,5 ) ))
```

Create a random, full-one and full-zero matrices of shape 2x2 :

```scala
rand(2,2)
ones(2,2)
zeros(2,2)
```

As a shortcut to `IMat(len, 1, Array(values))` to create a single-column matrix :

```scala
icol(1,2,3,4,5)
1 on 2 on 3 on 4 on 5
```

The same for single-row matrices :

```scala
irow(1,2,3,4)
1 \ 2 \ 3 \ 4
```

As with Mat, you'll find `col`, `dcol`, `cscol`, and `row`, `drow`, `csrow` types.

To get the diagonal : `getdiag(fmat)`

To make a square matrix with a given diagonal : `mkdiag(1 on 2)`

Element-wise matrix operations :

```scala
a + b // element-wise addition
a - b // element-wise subtraction
b ∘ a // (a *@ b) multiplication
b / a /* element wise-division */
```

Matrix operation

```scala
a.t // transpose
a * b // matrix multiplication
a ^* b  // transpose first matrix before multiplication
a *^ b  /* transpose second matrix before multiplication */
```

Dot products

```scala
a ∙ b // (a dot b) Column-wise dot product
a ∙→ b /* (a dotr b) Row-wise dot product */
```

Cartesian product

```scala
a ⊗ b /* (a kron b) Kronecker product */
```

Statistics per column :

```scala
sum(fmat)
mean(fmat)
variance(fmat)
maxi(fmat)
maxi2(fmat) // returns max and argmax
mini(fmat)
mini2(fmat) /* returns min and argmin */
```

Other operations per column :

```scala
cumsum(fmat)
sort(fmat) // sorted values
sortdown(fmat) // sorted values
sort2(fmat) // sorted values and indices
sortdown2(fmat) /* sorted values and indices */
```

The same statistics and operations per rows by adding a 2 param :

```scala
sum(fmat,2)
```

Have a look at unique(), unique3() and uniquerows().

Endly, to reshape, you can convert it to a n-dimension array, reshape it, and convert the result array back to a FMat :

```scala
FND(fmat).reshape(3,2).toFMat(3,2)
```


# More complicated matrices

Convert to a GPU matrix :

```scala
GMat(fmat)
GIMat(imat)
```

Create a generic type matrix :

```scala
var g:Mat = null
g = fmat
g = GMat(fmat)
```

Convert to sparse format :

```scala
val sfmat = SMat(fmat)
val sfmat = sparse(fmat)
SDmat(fmat) // sparse double
GSMat(fmat) /* sparse GPU */
```

To convert it back to a dense matrix :

```scala
full(sfmat)
```

Create a sparse matrix with value 5 at position (2,4) and value 6 at position (3,4) :

```scala
sparse(1\2,3\3, 5\6 ,  4,4)
```

Create an accumulator matrix of shape (3,3), by accumulating 5 and 4 on the same cell (3,2):

```scala
accum( (2\1) on (2\1), 5 on 4, 3,3)
```

Enable matrix caching :

```scala
Mat.useCache = true
```

There is also a great tool, the dictionaries, where counts is an optional argument :

```scala
val d = Dict(CSMat(1,2,Array("A","B")),IMat(1,2, Array(5,4)))
d(0) // value for index 0 : A
d("A") // index of A : 0
d.count("A") // counts : 5
d.count(0) // same
d.cstr // dictionary
d.counts // counts
d(IMat(1,2,Array(0,1))) // bulk retrieval
d(CSMat(1,2,Array("A","B)))
```


# File I/O


Save the matrix `fmat` as a text file :

```scala
saveFMat("t2.txt",fmat,1)
```

With gz compression :

```scala
saveFMat("t.gz",fmat)
```

With lz4 compression :

```scala
saveFMat("t.lz4",fmat)
```

To load the file :

```scala
loadFMat("t.lz4")
```

Have a look at load (and saveAs), loadLibSVM, loadIDX for HDF5, LibSVM and IDX format.

For efficient I/O, convert the string matrices to sparce matrices of bytes with :

```scala
SBMat(smat)
```

# Others

BIDMach offers also

- many random functions : randperm(), rand(), normrnd(),...

- a timer and flops counter : `tic` to start/reset timer, `flip` to start/reset timer and gflop counter, `toc` to get time since, `flop` to get time and flops counter, `gflop` for GPU

- cache management

- complex types

- plot possibilities :

```scala
plot( 1\2\3\4\5,3.2\2.1\1.3\4.2\(-1.2))
```

![]({{ site.url }}/img/bidmach_plot.png)

```scala
show(Image(IMat(2,2,Array(255,0,0,255)   ) kron ones(100,100) ))
```

![]({{ site.url }}/img/bidmach_image_show.png)

# Data sources

A data source is an iterator, let's create 5 data with their respective label (0), and pull the data iteratively 2 by 2 :

```scala
val dataimat = irow(1 to 5)
val labelimat = izeros(1,5)

val dopts = new MatSource.Options
dopts.batchSize = 2
val m = new MatSource(Array(dataimat, labelimat),dopts)
m.init
m.hasNext
val value = m.next
```

The output `m.next` is an array of 2 IMat, the first one is the data, the second the label. `dopts.what` will give you all options.

Option `dopts.sample` enables to sample of fraction of the data, and `opts.addConstFeat` adds a feature with constant value to 1 to the data for regression bias.

Data sources can also be used as a sink, to put back some data, such as the prediction value

```scala
dopts.putBack = 1  
val predimat = izeros(1,5)
val m = new MatSource(Array(dataimat, predimat),dopts)
m.init
val value = m.next
value(1)(1)=1
m.next
m.mats
```

Let's now go further with file data sources.

For the purpose, let's create 50 files of data, 50 files of labels, with random values for the explicative variables, and a weighted sum for the label, for the demonstration :

```scala
val params = drand(30,1)

0 until 50 map( i => {
  val fmat = FMat(drand(30,1000))
  saveMat("data%02d.fmat.lz4" format i, fmat);
  saveMat("label%02d.imat.lz4" format i, IMat(params ^* fmat) > 7)
})
```

Note the use of `until`, while `to` would create 51 values.

Check matrices are correctly saved with : `loadMat("data26.fmat.lz4")`. Now there are correctly saved, let's create a file data source :

```scala
val fopts = new FileSource.Options
fopts.fnames = List( {i:Int => {"data%02d.fmat.lz4" format i}}, {i:Int => {"label%02d.imat.lz4" format i}} )
fopts.nstart = 0
fopts.nend = 50
fopts.batchSize = 1000

val fs = FileSource(fopts)

fs.init
fs.hasNext
val value = fs.next
```


`fopts.order=1` randomize the order of the columns.

`fopts.what` give all set options.

BatchSize should be kept smaller than ncols for the files. Ideally a submultiple.

Lastly, have a look at SFileSource for sparse file data source.


# Run a Random Forest regressor


On the files created previously, let's launch the random forest regressor :

```scala
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
```

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

- in **local mode**, download [Joda-time](https://sourceforge.net/projects/joda-time/files/joda-time/), compile Spark for Scala 2.10 and  launch :

```bash
$SPARK_HOME/bin/spark-shell --jars \
~/technologies/aws-java-sdk-1.10.51/lib/aws-java-sdk-1.10.51.jar,\
../../technologies/BIDMach2/lib/BIDMat.jar,\
../../technologies/BIDMach2/BIDMach.jar,\
../../technologies/joda-time-2.4/joda-time-2.4.jar
```

Then you can import the required libraries :

```scala
import BIDMach.models.RandomForest
```

- as **cluster of GPU**, for example 2 g2.2xlarge executor instances with our [NVIDIA+CUDA+BIDMACH AMI for Spark](http://christopher5106.github.io/big/data/2016/01/27/two-AMI-to-create-the-fastest-cluster-with-gpu-at-the-minimal-engineering-cost-with-EC2-NVIDIA-Spark-and-BIDMach.html#cluster_of_gpu):

Launch the Spark Shell and be sure to have only 1 core per GPU on each executor :

```bash
sudo ./bin/spark-shell \
--master=spark://ec2-54-229-155-126.eu-west-1.compute.amazonaws.com:7077 \
--jars /home/ec2-user/BIDMach/BIDMach.jar,/home/ec2-user/BIDMach/lib/BIDMat.jar,/home/ec2-user/BIDMach/lib/jhdf5.jar,/home/ec2-user/BIDMach/lib/commons-math3-3.2.jar,/home/ec2-user/BIDMach/lib/lz4-1.3.jar,/home/ec2-user/BIDMach/lib/json-io-4.1.6.jar,/home/ec2-user/BIDMach/lib/jcommon-1.0.23.jar,/home/ec2-user/BIDMach/lib/jcuda-0.7.5.jar,/home/ec2-user/BIDMach/lib/jcublas-0.7.5.jar,/home/ec2-user/BIDMach/lib/jcufft-0.7.5.jar,/home/ec2-user/BIDMach/lib/jcurand-0.7.5.jar,/home/ec2-user/BIDMach/lib/jcusparse-0.7.5.jar \
--driver-library-path="/home/ec2-user/BIDMach/lib" \
--conf "spark.executor.extraLibraryPath=/home/ec2-user/BIDMach/lib"
```


<a name="spark_prepare_data" />

# Prepare the data with Spark

Spark is ideal to split very large data files into smaller parts that will be saved in BIDMach file format to feed the BIDMach file data sources.

Spark naturally splits the input file into parts for each job, that are accessible via **mapPartitions** methods, in order to execute a custom function on each split, as for example :

```scala
val file = sc.textFile("myfile.csv")
val header_line = file.first()
val tail_file = file.filter( _ != header_line)
val allData = tail_file.mapPartitionsWithIndex( upload_lz4_fmat_to_S3 )
allData.collect()
```

Adjusting the parallelism (number of partitions) will adjust the split size / number of lz4 files.

To build the custom *upload_lz4_fmat_to_S3* method, let's first create a function to upload to S3 :

```scala
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
```

The Amazon SDK looks for the credentials available in the different contexts.

Then, define a function to convert a line (String) of data from a CSV into an array of explicative features and the label :

```scala
def convert_line_to_Expl_Label_Tuple(line : String) : (Array[Float],Float) = {
  val values = line.split(";")

  // process your line HERE

  (expl, label)
}
```

Lastly, combine the functions to create  *upload_lz4_fmat_to_S3* method :

```scala
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
```


# Hyperparameter tuning using grid search with Spark

Hyperparameters are the parameters of the prediction model : number of trees, depth, number of bins... for example for a random forest regressor.

Grid search consists in computing the model for a grid of combinations for the parameters, for example

```scala
import BIDMat.{CMat, CSMat, DMat, Dict, FMat, FND, GMat, GDMat, GIMat, GLMat, GSMat, GSDMat, HMat, IDict, Image, IMat, LMat, Mat, SMat, SBMat, SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMach.models.RandomForest

val ndepths = icol(1, 2, 3, 4, 5)  // 5 values
val ntrees = icol(5, 10, 20)  // 3 values

val ndepthsparams = iones(ntrees.nrows, 1) ⊗ ndepths
val ntreesparams = ntrees ⊗ iones(ndepths.nrows,1)
val hyperparameters = ndepthsparams \ ntreesparams

val hyperparamSeq = for( i <- Range(0, hyperparameters.nrows) ) yield(hyperparameters(i,?))
val hyperparamRDD = sc.parallelize(hyperparamSeq,2)

hyperparamRDD.mapPartitionsWithIndex(  (index: Int, it: Iterator[BIDMat.IMat]) => {
  it.toList.map(x => {
    val params = drand(30,1)
    0 until 50 map( i => {
      val fmat = FMat(drand(30,1000))
      saveMat("data%02d.fmat.lz4" format i, fmat);
      saveMat("label%02d.imat.lz4" format i, IMat(params ^* fmat) > 7)
    })
    val (mm,opts) = RandomForest.learner("data%02d.fmat.lz4","label%02d.imat.lz4")
    opts.batchSize = 1000
    opts.nend = 50
    opts.depth =  x(0,0)
    opts.ncats = 2
    opts.ntrees = x(0,1)
    opts.impurity = 0
    opts.nsamps = 12
    opts.nnodes = 50000
    opts.nbits = 16
    opts.gain = 0.001f
    mm.train
    index + ": ndepth "+x(0,0) + " & ntrees "+x(0,1)
    } ).iterator
  }).collect
```

which should give you :

    res17: Array[String] = Array(0: ndepth 1 & ntrees 5, 0: ndepth 2 & ntrees 5, 0: ndepth 3 & ntrees 5, 0: ndepth 4 & ntrees 5, 0: ndepth 5 & ntrees 5, 0: ndepth 1 & ntrees 10, 0: ndepth 2 & ntrees 10, 1: ndepth 3 & ntrees 10, 1: ndepth 4 & ntrees 10, 1: ndepth 5 & ntrees 10, 1: ndepth 1 & ntrees 20, 1: ndepth 2 & ntrees 20, 1: ndepth 3 & ntrees 20, 1: ndepth 4 & ntrees 20, 1: ndepth 5 & ntrees 20)

The hyperparameters have been distributed across the cluster with a RDD (Spark resilient dataset) :

    hyperparamRDD: org.apache.spark.rdd.RDD[BIDMat.IMat] = ParallelCollectionRDD[1] at parallelize at <console>:40

It is a very simple example, to show how to set up a cluster of GPU powered by BIDMach, but a normal hyperparameter tuning would evaluate the results for each set of hyperparameter and push them in the result RDD.

In the directory */home/ec2-user/spark-1.6.0/work/app-20160215174421-0026/0/*, I get all the temporary files linked to the job. Up to you to set all that in a more appropriate directory such as `/mnt`, `/tmp` ... ! My demo is done.

**Well done!**
