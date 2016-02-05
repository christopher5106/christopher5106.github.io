---
layout: post
title:  "Big data tutorial on BIDMach library : basic matrix operations and file IO. Example on a RandomForest computation"
date:   2016-02-05 23:00:51
categories: big data
---

BIDMach is a powerful library,

- on matrix operations, as powerful as Python Numpy library on CPU, and offering the fastest GPU equivalent.

- on file i/o, in particular with [lz4 compression](https://github.com/Cyan4973/lz4).

Let's put things in order :)

# Basic matrices

Let's create our first matrix of integers of size 2x2, then a matrix of floats of size 3x2, a matrix of double values of size 2x2 and a matrix of strings :

    val imat = IMat(2,2, Array(1,2,3,4))
    val fmat = FMat(2,3, Array(1,2,3,4,5,6))
    val dmat = DMat(2,2)
    val smat = CSMat(1,3, Array("you","and","me"))

Access the number of columns, rows :

    size(fmat)
    fmat.dims
    fmat.length
    fmat.ncols
    fmat.nrows

Two ways to access element on (row 1, column 2), either by *row x column* tuple, or by *index* (column-oriented numbering) :

    fmat(0,1)
    fmat(2)

Access all elements as one column :

    fmat(?)

Access first column and first row :

    fmat(?,0)
    fmat(0,?)

Access elements with indexes between 1 and 3 (not inclusive):

    fmat(1->3)

Create a Matrix of size 2x2, with 3rd element on first position, 4th on second, 2nd on third position and 5th on last position :

    fmat(IMat(2,2,Array( 2,3,1,5 ) ))

Create a random, full-one and full-zero matrices of shape 2x2 :

    rand(2,2)
    ones(2,2)
    zeros(2,2)

As a shortcut to `IMat(len, 1, Array(values))` to create a single-column matrix :

    icol(1,2,3,4,5)
    1 on 2 on 3 on 4 on 5

The same for single-row matrices :

    irow(1,2,3,4)
    1 \ 2 \ 3 \ 4

As with Mat, you'll find `col`, `dcol`, `cscol`, and `row`, `drow`, `csrow` types.

To get the diagonal : `getdiag(fmat)`

To make a square matrix with a given diagonal : `mkdiag(1 on 2)`

Element-wise matrix operations :

    a + b element-wise addition
    a - b element-wise subtraction
    b ∘ a (a *@ b) multiplication
    b / a element wise-division

Matrix operation

    a.t transpose
    a * b matrix multiplication
    a ^* b  transpose first matrix before multiplication
    a *^ b  transpose second matrix before multiplication

Dot products

    a ∙ b (a dot b) Column-wise dot product
    a ∙→ b (a dotr b) Row-wise dot product

Cartesian product

    a ⊗ b (a kron b)       Kronecker product

Statistics per column :

    sum(fmat)
    mean(fmat)
    variance(fmat)
    maxi(fmat)
    maxi2(fmat) # returns max and argmax
    mini(fmat)
    mini2(fmat) # returns min and argmin

Other operations per column :

    cumsum(fmat)
    sort(fmat) #sorted values
    sortdown(fmat) #sorted values
    sort2(fmat) #sorted values and indices
    sortdown2(fmat) #sorted values and indices

The same statistics and operations per rows by adding a 2 param :

    sum(fmat,2)

Have a look at unique(), unique3() and uniquerows().

Endly, to reshape, you can convert it to a n-dimension array, reshape it, and convert the result array back to a FMat :

    FND(fmat).reshape(3,2).toFMat(3,2)


# More complicated matrices

Convert to a GPU matrix :

    GMat(fmat)
    GIMat(imat)

Create a generic type matrix :

    var g:Mat = null
    g = fmat
    g = GMat(fmat)

Convert to sparse format :

    val sfmat = SMat(fmat)
    val sfmat = sparse(fmat)
    SDmat(fmat) # sparse double
    GSMat(fmat) # sparse GPU

To convert it back to a dense matrix :

    full(sfmat)

Create a sparse matrix with value 5 at position (2,4) and value 6 at position (3,4) :

    sparse(1\2,3\3, 5\6 ,  4,4)

Create an accumulator matrix of shape (3,3), by accumulating 5 and 4 on the same cell (3,2):

    accum( (2\1) on (2\1), 5 on 4, 3,3)

Enable matrix caching :

    Mat.useCache = true

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
