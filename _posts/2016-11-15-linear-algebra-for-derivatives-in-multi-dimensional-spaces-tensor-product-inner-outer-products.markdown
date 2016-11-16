---
layout: post
title:  "Linear algebra for derivatives in multi-dimensional spaces : tensor products, inner and outer products..."
date:   2016-11-15 11:00:51
categories: deep learning
---

Deep learning involves lot's of the derivatives'computations in multi-dimensional spaces.

In particular, any network module, as well as a full network composed of multiple modules, are mapping functions between input and output spaces

$$ \mathbb{R}^m \rightarrow \mathbb{R}^o $$

but also between parameter and output space

$$ \mathcal{N} : \mathbb{R}^n \rightarrow \mathbb{R}^o $$

which is of much interest for the gradient descents.

A loss function is a mapping to a 1-dimension space (scalar) :

$$ \mathcal{L} : \mathbb{R}^o \rightarrow \mathbb{R} $$

# Linear algebra

A matrix multiplication (or tensor product in higher dimensions) gives a new matrix :

$$ A \times B = [ \sum_k a_{i,k} b_{k,j} ]_{i,j} $$

The inner product or scalar product or of 2 vectors outputs a scalar :

$$ u \cdot v = \sum_i u_i v_i = u^T \times v $$

The same for matrices, the matrix dot product or Frobenius inner product, outputs a scalar :

$$ A \odot B = \sum_{i,j} a_{i,j} b_{i,j} $$

The outer product of 2 vectors produces a matrix :

$$ u \otimes v = u \times v^T = [ u_i v_j ]_{i,j}  $$


# Jacobian

The Jacobian is the first order derivative of the function.

For the network module, the Jacobian is a matrix $$ \in \mathbb{R}^{o \times n} $$

$$ J_{i,j} = \frac{ \partial \mathcal{N_i} }{ \partial w_j } $$

$$ J = \frac{ \partial \vec{\mathcal{N}} }{ \partial \vec{w}^T } $$

where j is the indice of the output in the network output and $$ \vec{w} $$ is a column vector.

In the case of the loss function, the Jacobian is usually the vector

$$ j_{i} = \frac{ \partial \mathcal{L} }{ \partial u_i }$$

$$ j =  \frac{ \partial \mathcal{L} }{ \partial \vec{u} } $$

but I'll write it as a matrix with one row $$ \in \mathbb{R}^{1 \times o} $$

$$ J_{0,i} = \frac{ \partial \mathcal{L} }{ \partial u_i } $$

$$ J = \frac{ \partial \mathcal{L} }{ \partial \vec{u}^T } $$

where i is the indice of the output in the network output.

# Hessian

The hessian is the second order derivative, following the same definition as for Jacobian :

$$ H_{\mathcal{N}} = \frac{ \partial J_{\mathcal{N}} }{ \partial \vec{w}^{T2} } = \frac{ \partial^2 \vec{\mathcal{N}} }{ \partial \vec{w}^T \partial \vec{w}^{T2} } $$

Since $$ J_{\mathcal{N}}$$ is a matrix, $$ H_{\mathcal{N}} $$ is a 3-dimensional tensor. T2 is for the transpose to the third dimension.

$$ (H_{\mathcal{N}})_{i,j,k} = \frac{ \partial^2 \mathcal{N}_i }{ \partial w_j \partial w_k } $$


Let's write the special case for a scalar function, for which the hessian is a (o x o) symetric matrix

$$ h_{\mathcal{L}} = \frac{ \partial^2 \mathcal{L}}{ \partial \vec{w} \partial \vec{w}^T  } = \frac{ \partial \vec{j_{\mathcal{L}}} }{ \partial \vec{w}^T } $$

$$ (h_{\mathcal{L}})_{i,j} = \frac{ \partial^2 \mathcal{L} }{ \partial w_i \partial w_j } $$


# Composition of functions

A composition is for example the loss function computed on the output of the network (softmax can be seen as a module inside the network or inside the loss function)

$$ \mathcal{C} = \mathcal{L} \circ \mathcal{N} $$

In the general case when $$ \mathcal{L} $$ has a multi-dimensional output

$$ \frac{ \partial \mathcal{C}_i }{ \partial w_j } = \sum_k \frac{ \partial \mathcal{L}_i }{ \partial u_k } \times \frac{ \partial \mathcal{N}_k }{ \partial w_j } $$

which is a simple matrix multiplication

$$ J_{\mathcal{C}} =  J_{\mathcal{L}} \times  J_{\mathcal{N}} $$

And in the scalar case (when $$ \mathcal{L} $$ outputs a scalar), this can be rewritten with the vector notation :

$$ j_{\mathcal{C}} =  J_{\mathcal{N}}^T \times j_{\mathcal{L}} $$

That is why it can sometimes be a bit confusing.

Let's go for the hessian of a composition of functions, but considering the scalar case only (the multi-dimensional case is left as an exercice for the reader :) ), let's keep in mind that the jacobian of the loss function is being evaluated at the output of the network in fact :


$$ \frac{ \partial \mathcal{C} }{ \partial w_j } = \sum_k  \left( \frac{ \partial \mathcal{L} }{ \partial u_k } \circ \mathcal{N}  \right) \times \frac{ \partial \mathcal{N}_k }{ \partial w_j } $$

and derivate (if you have a headache, it might not be an anomaly) :

$$ \frac{ \partial \mathcal{C} }{ \partial w_i \partial w_j } =  \sum_k \sum_l \frac{ \partial^2 \mathcal{L} }{ \partial u_k \partial u_l } \times \frac{ \partial \mathcal{N}_l}{\partial w_i}  \times \frac{ \partial \mathcal{N}_k }{ \partial w_j } + \sum_k  \frac{ \partial \mathcal{L} }{ \partial u_k } \times \frac{ \partial \mathcal{N}_k }{ \partial w_i \partial w_j } $$


$$ h_{\mathcal{C}} = J_{\mathcal{N}}^T \times h_{\mathcal{L}} \times J_{\mathcal{N}} + \sum_k (J_{\mathcal{L}})_k \times h_{\mathcal{N}_k} $$

The first part $$ J_{\mathcal{N}}^T \times h_{\mathcal{L}} \times J_{\mathcal{N}} $$ is the **Gauss-Newton matrix**, modeling the interactions of second order originated from the top part $$ \mathcal{L} $$. It is positive semi definite and is used as an approximation in some second order optimization algorithms. 


# Matching loss function

Let's consider the case where $$ \mathcal{N} $$ is the output non-linearity module, such as softmax.

It is easy to see that

$$ \frac{ \partial \mathcal{N}_i }{ \partial u_j } = \mathcal{N}_i \times (1_{i == j} - \mathcal{N}_j) $$

Hence, for the loss function $$ \mathcal{L} = Y \cdot \log \mathcal{N} $$ where Y is the one-hot encoding of the correct label :

$$ \frac{ \partial \mathcal{L} \circ \mathcal{N} }{ \partial \vec{u} } = Y - \mathcal{N} $$

We say **the log likelihood loss function matches the softmax output non-linearity** since its Jocabian is an affine transformation of the output.

In the same way, **the mean squared error loss matches a linear output module**.
