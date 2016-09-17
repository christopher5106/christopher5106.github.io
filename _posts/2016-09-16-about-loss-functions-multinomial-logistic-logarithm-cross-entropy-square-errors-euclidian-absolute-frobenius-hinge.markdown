---
layout: post
title:  "About loss functions : multinomial logistic, cross entropy, square errors, euclidian, hinge, Crammer and Singer, one versus all, squared hinge, absolute value, L1 / L2 / Frobenius norms"
date:   2016-09-16 17:00:51
categories: deep learning
---

In machine learning many different losses exist.

A loss is a "penalty" score to reduce with the training of the algorithm on data. It is usually called the **objective function** to optimize.

Let's remind the different loss for the different cases. Knowing each of them will help the datascientists choose the right one for his problem.

Let's remind a few concepts.


# The likelihood and the log loss

When we see a variable X depending of another variable $$ \theta $$, the **likelihood** is the function in $$ \theta $$

$$ f( \theta ) = P(X | \theta ) $$

and the log-likelihood which is more convenient to work with is :

$$ loglike( \theta ) = \ln P(X | \theta ) $$

The log-likelihood of statistically independant observation will simply be the sum of the log-likelihood of each observation.

We usually want to estimate the parameter $$ \theta $$ and that's why we try to maximise the log-likelihood (which is equivalent to maximising the likelihood itself). In the case of independant observations, it is equivalent to maximising the log-likelihood of each observation.

Last, as an *objective function or a loss*, we usually prefer to say that we "minimize" it, that's why the **negative log likelihood** is used :

$$ negloglike( \theta ) = - \ln P(X | \theta ) $$


# Binomial probabilities - log loss / logistic loss / cross-entropy loss

Binomial means 2 classes, which are usually 0 or 1.

Each class has a probability $$ p $$ and $$ 1 - p $$ (sums to 1).

When using a network, we try to get 0 and 1 as values, that's why we add a **sigmoid function or logistic function** that saturates as a last layer :

![]({{site.url}}/img/sigmoid.png)

$$ f:  x \rightarrow \frac{1}{ 1 + \exp^{-x}} $$


Then, once the estimated probability to get 1 is $$ y_p $$, then it is easy to see that the negative log likelihood can be written

$$ negloglike( y ) = - y \log y_p  - (1 - y) \log ( 1 - y_p ) $$

which is also the **cross-entropy**

$$ crossentropy (p , q ) = E_p [ -\log q ] = - \sum_x p(x ) \log q(x) $$

In information theory, if you try to identify all classes with a code of a length depending of his probability, that is $$ \log q $$, where q is your estimated probability, then the expected length in reality is given by the cross-entropy.

Last, let's remind that the combined sigmoid and cross-entropy has a very simple and stable derivative.


# Multinomial probabilities / multi-class classification : multinomial logistic loss / cross entropy loss / log loss


It is a problem where we have *k* classes or categories, and only one valid for each example.

The target values are still binary but represented as a vector y that will be defined by the following if the example x is of class c :

$$  y_i =   \begin{cases}
      0, & \text{if}\ i \neq c \\
      1, & \text{otherwise}
    \end{cases}
$$

If $$ \{ p_i \} $$ is the probability of each class, then it is a multinomial distribution and

$$ \sum_i p_i = 1 $$

The equivalent to the sigmoid function in multi-dimensional space is the **softmax function or logistic function or normalized exponential function** to produce such a distribution from any input :

$$ z \rightarrow \frac{\exp z_i }{ \sum_k \exp^{z_k} }  $$


The error is also best described by cross-entropy :

$$ l(y) = - \sum_{i=0}^k y_i \ln p_i $$

Cross-entropy is designed to deal with errors on probabilities. For example, $$ \ln(0.01) $$ will be a lot stronger error signal than $$ \ln(0.1) $$ and encourage to resolve errors. In some cases, the logarithm is bounded to avoid extreme punishments.

In information theory, optimizing the log loss means minimizing the description length of y. The information content of a random variable to be in the class being $$ - \ln p_i $$.

Last, let's remind that the combined softmax and cross-entropy has a very simple and stable derivative.


# Multi-label classification

There is a variant for multi-label classification, in this case multiple $$ y_i $$ can have a value set to 1.

For example, "car", "automotible", "motor vehicule" are three labels that can be applied to a same image of a car. On the image of a truck, you'll only have "motor vehicule" active for example.

In this case, the softmax function will not apply, we usually add a sigmoïd layer before the cross-entropy layer to ensure stable gradient estimation :

$$ \frac{1}{1+\exp^{-t}} $$

The cross-entropy will look like :

$$ l(y) = - \sum_{i=0}^k y_i \ln p_i + (1 - y_i) \ln ( 1 - p_i ) $$



# Square error / Sum of squares / Euclidian loss


This time, contrary to previous estimations that were probabilities, when predictions are scalars or metrics, we usually use the **square error or euclidian loss** which is the L2-norm of the error :

$$ l(y) = \sum_i ( ŷ_i - y_i )^2 $$

Minimising the squared error is equivalent to predicting the (conditional) mean of y.

Due to the gradient being flat at the extremes for a sigmoid function, we do not use a sigmoid activation with a squared error loss because convergence will be slow if some neurons saturate on the wrong side.

A squared error is often used with a rectified linear unit.


# Absolute value loss

The absolute value loss is the L1-norm of the error :

$$ l(y) = \sum_i |  ŷ_i - y_i | $$

Minimizing the absolute value loss means predicting the (conditional) median of y. Variants can handle other quantiles. 0/1 loss for classification is a special case.


# Hinge loss / Maximum margin

Hinge loss is trying to separate the positive and negative examples $$ (x,y) $$, x being the input, y the target $$ \in \{-1, 1 \} $$, the loss for a linear model is defined by

$$ l(y) = \max (0, 1 - y w \cdot x ) $$

The minimization of the loss will only consider examples that infringe the margin, otherwise the gradient will be zero since the max saturates.

In order to minimize the loss,

- positive example will have to output a result superior to 1 :  $$ w \cdot x > 1$$

- negative example will have to output a result inferior to -1 : $$ w \cdot x < - 1$$


# Crammer and Singer loss / One-versus-All Hinge loss


Crammer and Singer defined a multi-class version of the hinge loss :

$$ l(y) = \max (0, 1 + \max_{ c \neq y }  w_c \cdot x - w_y \cdot x ) $$

so that minimizing the loss means to do both :

- maximize the prediction $$ w_y \cdot x $$ of the correct class

- minimize the predicted value $$ w_c \cdot x $$ for all other classes c that have the maximal value

until for all these other classes, their predicted values $$ w_c \cdot x $$ are all below $$ w_y \cdot x -1 $$, the value for the correct class with a margin of 1.


#  Squared hinge loss

It is simply the square of the hinge loss :

$$ l(y) = \max (0, 1 - y w \cdot x )^2 $$


# Infogain loss


# L1 / L2 / Frobenius norms
