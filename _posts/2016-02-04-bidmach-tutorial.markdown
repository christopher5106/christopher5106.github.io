---
layout: post
title:  "Big data tutorial on BIDMach library : basic matrix operations and file I/O. Example on a RandomForest computation"
date:   2016-02-04 23:00:51
categories: big data
---

BIDMach is a very powerful library,

- on matrix operations, as powerful as Python Numpy library on CPU, but offering to possibility to work on CPU and GPU indifferently.

- on file i/o, in particular with [lz4 compression](https://github.com/Cyan4973/lz4).

Let's put things back in order :)

First, we'll see basic matrices, then more complicated types, then how to prepare data and an example of random forest computation.

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
b / a // element wise-division
{% endhighlight %}

Matrix operation

{% highlight scala %}
a.t // transpose
a * b // matrix multiplication
a ^* b  // transpose first matrix before multiplication
a *^ b  // transpose second matrix before multiplication
{% endhighlight %}

Dot products

{% highlight scala %}
a ∙ b // (a dot b) Column-wise dot product
a ∙→ b // (a dotr b) Row-wise dot product
{% endhighlight %}

Cartesian product

{% highlight scala %}
a ⊗ b // (a kron b) Kronecker product
{% endhighlight %}

Statistics per column :

{% highlight scala %}
sum(fmat)
mean(fmat)
variance(fmat)
maxi(fmat)
maxi2(fmat) // returns max and argmax
mini(fmat)
mini2(fmat) // returns min and argmin
{% endhighlight %}

Other operations per column :

{% highlight scala %}
cumsum(fmat)
sort(fmat) // sorted values
sortdown(fmat) // sorted values
sort2(fmat) // sorted values and indices
sortdown2(fmat) // sorted values and indices
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
GSMat(fmat) // sparse GPU
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
d(0) # value for index 0 : A
d("A") # index of A : 0
d.count("A") # counts : 5
d.count(0) # same
d.cstr // dictionary
d.counts // counts
d(IMat(1,2,Array(0,1))) // bulk retrieval
d(CSMat(1,2,Array("A","B)))
{% endhighlight %}


# File I/O


Save the matrix `fmat` as a text file :

    saveFMat("t2.txt",fmat,1)

With gz compression :

    saveFMat("t.gz",fmat)

With lz4 compression :

    saveFMat("t.lz4",fmat)

To load the file :

    loadFMat("t.lz4")

Have a look at load (and saveAs), loadLibSVM, loadIDX for HDF5, LibSVM and IDX format.

For efficient I/O, convert the string matrices to sparce matrices of bytes with :

    SBMat(smat)

# Others

BIDMach offers also

- many random functions : randperm(), rand(), normrnd(),...

- a timer and flops counter : `tic` to start/reset timer, `flip` to start/reset timer and gflop counter, `toc` to get time since, `flop` to get time and flops counter, `gflop` for GPU

- cache management

- complex types

- plot possibilities :

        plot( 1\2\3\4\5,3.2\2.1\1.3\4.2\(-1.2))


![]({{ site.url }}/img/bidmach_plot.png)


    show(Image(IMat(2,2,Array(255,0,0,255)   ) kron ones(100,100) ))


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

For the purpose, let's create 50 files of data, 50 files of labels, with random values each for the demonstration :

{% highlight scala %}
0 to 50 map( i => {
   saveMat("data%02d.dmat.lz4" format i ,drand(2,100));
   saveMat("label%02d.imat.lz4" format i, IMat(drand(1,100) > 0.5))
})
{% endhighlight %}


Check matrices are correctly saved with : `loadMat("data26.fmat.lz4")`. Now there are correctly saved, let's create a file data source :

{% highlight scala %}
val fopts = new FileSource.Options
fopts.fnames = List( {i:Int => {"data%02d.dmat.lz4" format i}}, {i:Int => {"label%02d.imat.lz4" format i}} )
fopts.nstart = 0
fopts.nend = 5001
fopts.batchSize = 100

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
val (mm,opts) = RandomForest.learner("data%02d.dmat.lz4","label%02d.imat.lz4")

opts.batchSize = 100
opts.nend = 5001

opts.depth =  24
opts.ntrees = 100
opts.impurity = 0
opts.nsamps = 12
opts.nnodes = 50000
opts.nbits = 16
opts.gain = 0.001f

mm.train
{% endhighlight %}


# Run BIDMach on Spark local mode

Instead of running `bidmach` command to launch a bidmach shell, let's run the same commands inside Spark in local mode, adding the BIDMat and BIDMach libraries to the classpath :

    ./bin/spark-shell --jars ../BIDMat/BIDMat.jar,../BIDMach/BIDMach.jar

 and import the required libraries :

    import BIDMach.models.RandomForest


# Prepare the data and launch an hyperparameter tuning with Spark

Let's launch a cluster with 1 master and 2 g2.2xlarge instances with our [NVIDIA+CUDA+BIDMACH AMI for Spark](http://christopher5106.github.io/big/data/2016/01/27/two-AMI-to-create-the-fastest-cluster-with-gpu-at-the-minimal-engineering-cost-with-EC2-NVIDIA-Spark-and-BIDMach.html):

    ./ec2/spark-ec2 -k sparkclusterkey -i ~/sparkclusterkey.pem --region=eu-west-1 --copy-aws-credentials --instance-type=g2.2xlarge -s 2 --hadoop-major-version=2 --spark-ec2-git-repo=https://github.com/christopher5106/spark-ec2 launch spark-cluster


And log in

    ./ec2/spark-ec2 -k sparkclusterkey -i ~/sparkclusterkey.pem --region=eu-west-1 login spark-cluster


Launch the Spark Shell and be sure to have only 1 core per GPU on each executor :

    ./spark/bin/spark-shell --conf spark.executor.cores=1 --jars /home/ec2-user/BIDMat/BIDMat.jar,/home/ec2-user/BIDMach/BIDMach.jar

Prepare your data with a first Spark job : Spark `saveAsTextFile` method is ideal to prepare data files for the BIDMach file data sources.

    sc.hadoopConfiguration.set("fs.s3n.awsAccessKeyId", "XXX")
    sc.hadoopConfiguration.set("fs.s3n.awsSecretAccessKey","YYY")
    val file = sc.textFile("s3n://BUCKET/FILE.csv")

    # remove header line
    val header_line = file.first()
    val tail_file = file.filter( _ != header_line)

    data = tail_file.map( line => {
      # proceed data to create a column of features
    })
    label = tail_file.map( line => {
      # proceed data to create a column 1-hot encoding of the label
    })

    import org.apache.hadoop.io.compress.GzipCodec
    data.saveAsTextFile("s3n://BUCKET/data", classOf[GzipCodec])
    label.saveAsTextFile("s3n://BUCKET/label", classOf[GzipCodec])

will create a list of compressed files named in the format **data/part-%05d.gz** and **label/part-%05d.gz**.

Let's launch a hyperparameter tuning job using grid search.
