---
layout: post
title:  "About loss functions, regularization and joint losses : multinomial logistic, cross entropy, square errors, euclidian, hinge, Crammer and Singer, one versus all, squared hinge, absolute value, infogain, L1 / L2 - Frobenius / L2,1 norms, connectionist temporal classification loss"
date:   2016-09-16 17:00:51
categories: deep learning
---

In machine learning many different losses exist.

A loss is a "penalty" score to reduce when training an algorithm on data. It is usually called the **objective function** to optimize.

Let's remind the different losses for the different cases. Knowing each of them will help the datascientists choose the right one for his problem.


# The likelihood and the log loss

Classifying means assigning a label to an observation :

$$ x \rightarrow y $$

Such a function is named a **classifier**. To create such classifier, we usually create models with parameters to define :

$$ f_w : x \rightarrow y $$

The process of defining the optimal parameters w given past observations X and their known labels Y is named **training**. The objective of the training is obviously to maximise the **likelihood**

$$ \text{likelihood}( w ) = P_w( y | x ) $$

Since the logarithm is monotonous, it is equivalent to **minimize the negative log-likelihood** :

$$ \mathscr{L}( w ) = - \ln P_w( y | x ) $$

The reason for taking the negative of the logarithm of the likelihood are

- it is more convenient to work with the log, because the log-likelihood of statistically independant observation will simply be the sum of the log-likelihood of each observation.

- we usually prefer to write the **objective function** as a **cost function** to minimize.


# Binomial probabilities - log loss / logistic loss / cross-entropy loss

Binomial means 2 classes, which are usually 0 or 1.

Each class has a probability $$ p $$ and $$ 1 - p $$ (sums to 1).

When using a network, we try to get 0 and 1 as values, that's why we add a **sigmoid function or logistic function** that saturates as a last layer :

![]({{ site.url }}/img/sigmoid.png)

$$ f:  x \rightarrow \frac{1}{ 1 + e^{-x}} $$


Then, once the estimated probability to get 1 is $$ \hat{p}  $$ (sometimes written $$ ŷ $$ also), then it is easy to see that the negative log likelihood can be written

$$ \mathscr{L} = - y \log \hat{p}  - (1 - y) \log ( 1 - \hat{p} ) $$

which is also the **cross-entropy**

$$ \text{crossentropy} (p , q ) = E_p [ -\log q ] = - \sum_x p(x ) \log q(x) = - \frac{1}{N} \sum_{n=1}^N \log q(x_n) $$

In information theory, if you try to identify all classes with a code of a length depending of his probability, that is $$ - \log q $$, where q is your estimated probability, then the expected length in reality (p being the real probability) is given by the cross-entropy.

Note that in natural language processing, we may also speek of **perplexity** defined by

$$ 2^{ \text{crossentropy}(p,q)} = 2^{ - \frac{1}{N} \sum_{n=1}^N \log_2 q(x_n) } $$

seen as an indicator of the number of possibilities for the variable to predict, since the perplexity of a uniform k-class random variable would be k.

Last, let's remind that the combined sigmoid and cross-entropy has a very simple and stable derivative

$$ \hat{p} - y $$


NB: if you choose that your labels $$ y \in \{ \pm 1 \} $$, you can write the binary logistic loss

$$ \mathscr{L} = \log \left(1 + e^{-y \cdot \hat{p} } \right) $$

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

The equivalent to the sigmoid function in multi-dimensional space is the **softmax function or logistic function or normalized exponential function** to produce such a distribution from any input vector z :

$$ z \rightarrow \left\{ \frac{\exp z_i }{ \sum_k \exp^{z_k} } \right\}_i  $$


The error is also best described by cross-entropy :

$$ \mathscr{L} = - \sum_{i=0}^k y_i \ln \hat{p}_i $$

Cross-entropy is designed to deal with errors on probabilities. For example, $$ \ln(0.01) $$ will be a lot stronger error signal than $$ \ln(0.1) $$ and encourage to resolve errors. In some cases, the logarithm is bounded to avoid extreme punishments.

In information theory, optimizing the log loss means minimizing the description length of y. The information content of a random variable to be in the class being $$ - \ln p_i $$.

Last, let's remind that the combined softmax and cross-entropy has a very simple and stable derivative.


# Multi-label classification

There is a variant for multi-label classification, in this case multiple $$ y_i $$ can have a value set to 1.

For example, "car", "automotible", "motor vehicule" are three labels that can be applied to a same image of a car. On the image of a truck, you'll only have "motor vehicule" active for example.

In this case, the softmax function will not apply, we usually add a sigmoïd layer before the cross-entropy layer to ensure stable gradient estimation :

$$ t \rightarrow \frac{1}{1+e^{-t}} $$

The cross-entropy will look like :

$$ \mathscr{L} = - \sum_{i=0}^k y_i \ln \hat{p}_i + (1 - y_i) \ln ( 1 - \hat{p}_i ) $$


# Infogain loss / relative entropy

Many synonym exists : Kullback–Leibler divergence, discrimination information, information divergence, information gain, relative entropy, KLIC, KL divergence.

It measures the difference between two probabilities.

$$ KL(p,q) = H(p,q) - H(p) = - \sum p(x) \log q(x) + \sum p(x) \log p(x) = \sum p(x ) \ln \frac{ p(x)}{q(x)} $$

hence in our nomenclature :

$$ \mathscr{L} = \sum_i y_i \ln \frac{ y_i }{ \hat{p}_i } $$

The infogain is the difference between the entropy before and the entropy after.

# Square error / Sum of squares / Euclidian loss


This time, contrary to previous estimations that were probabilities, when predictions are scalars or metrics, we usually use the **square error or euclidian loss** which is the L2-norm of the error :

$$ \mathscr{L} = \sum_i ( ŷ_i - y_i )^2 = \| ŷ - y \|_2^2 $$

Minimising the squared error is equivalent to predicting the (conditional) mean of y.

Due to the gradient being flat at the extremes for a sigmoid function, we do not use a sigmoid activation with a squared error loss because convergence will be slow if some neurons saturate on the wrong side.

A squared error is often used with a rectified linear unit.

The L2 norm penalizes large errors more strongly and therefore is very sensitive to outliers. To avoid this, we usually use the squared root version :

$$ \mathscr{L} = \| ŷ - y \|_2 $$



# Absolute value loss / L1 loss

The absolute value loss is the L1-norm of the error :

$$ \mathscr{L} = \sum_i |  ŷ_i - y_i | = \| ŷ - y \|_1 $$

Minimizing the absolute value loss means predicting the (conditional) median of y. Variants can handle other quantiles. 0/1 loss for classification is a special case.

Note that the L1 norm is not differentiable in 0, and it is possible to use a smooth L1 :

$$ | d |_{\text{smooth}} = =   \begin{cases}
      0.5 d^2, & \text{if}\ | d  | \leq 1 \\
      | d | - 0.5, & \text{otherwise}
    \end{cases}
    $$

Although the L2 norm is more precise and better in minizing prediction errors, the L1 norm produces sparser solutions, ignore more easily fine details and is less sensitive to outliers. Sparser solutions are good for feature selection in high dimensional spaces, as well for prediction speed.


# Hinge loss / Maximum margin

Hinge loss is trying to separate the positive and negative examples $$ (x,y) $$, x being the input, y the target $$ \in \{-1, 1 \} $$, the loss for a linear model is defined by

$$ \mathscr{L}(w) = \max (0, 1 - y w \cdot x ) $$

The minimization of the loss will only consider examples that infringe the margin, otherwise the gradient will be zero since the max saturates.

In order to minimize the loss,

- positive example will have to output a result superior to 1 :  $$ w \cdot x > 1$$

- negative example will have to output a result inferior to -1 : $$ w \cdot x < - 1$$

The hinge loss is a convex function, easy to minimize. Although it is not differentiable, it's easy to compute its gradient locally. There exists also [a smooth version of the gradient](https://en.wikipedia.org/wiki/Hinge_loss).

#  Squared hinge loss

It is simply the square of the hinge loss :

$$ \mathscr{L}(w) = \max (0, 1 - y w \cdot x )^2 $$


# One-versus-All Hinge loss

The multi-class version of the hinge loss

$$ \mathscr{L}(w) = \sum_c \max (0, 1 -  \mathbb{1}_{y,c} w \cdot x ) $$

where

$$   \mathbb{1}_{y,c} =   \begin{cases}
      -1, & \text{if}\ y \neq c \\
      1, & \text{otherwise}
    \end{cases}
$$


# Crammer and Singer loss


Crammer and Singer defined a multi-class version of the hinge loss :

$$ \mathscr{L}(w) = \max (0, 1 + \max_{ c \neq y }  w_c \cdot x - w_y \cdot x ) $$

so that minimizing the loss means to do both :

- maximize the prediction $$ w_y \cdot x $$ of the correct class

- minimize the predicted value $$ w_c \cdot x $$ for all other classes c that have the maximal value

until for all these other classes, their predicted values $$ w_c \cdot x $$ are all below $$ w_y \cdot x -1 $$, the value for the correct class with a margin of 1.

Note that is possible to replace the 1 with a smooth $$ \Delta ( y, c) $$ value that measures the dissimilarity :

$$ \mathscr{L}(w) = \max (0, \max_{ c \neq y } \Delta ( y, c) + w_c \cdot x - w_y \cdot x ) $$



# L1 / L2, Frobenius / L2,1 norms

It is frequent to add some regularization terms to the cost function

$$ \text{min}_w \mathscr{L}(w) + \gamma R(w) $$

such as

- the L1-norm, for the LASSO regularization

$$ \| w \|_1 = \sum_{i,j} | w_{i,j} | = \sum_i \| w_i \|_1 $$

- the L2-norm or Frobenius norm, for the ridge regularization

$$ \| w \|_2 = \sqrt{ \sum_{i,j} w_{i,j}^2 } = \sqrt{ \sum_i  \| w_i \|_2^2 } $$

- the L2,1 norm, used for [discriminative feature selection](https://www.aaai.org/ocs/index.php/IJCAI/IJCAI11/paper/viewFile/3136/3481)

$$ \| w \|_{2,1} = \sum_i \sqrt{ \sum_j w_{i,j}^2 } $$

# Joint embedding

A **joint loss** is a sum of two losses :

$$ \text{min}_{w_1,w_2} \mathscr{L}_1(w_1) + \mathscr{L}_2(w_2) $$

and in the case of multi-modal classification, where data is composed of multiple parts, such as for example images (x1) and texts (x2), we usually use the joint loss with multiple **embeddings**, which are high dimensional feature spaces :

$$ f_{w_1} : x_1 \rightarrow z_1 $$

$$ g_{w_2} : x_2 \rightarrow z_2 $$

and a similarity function, such as for example,

$$ s_{w_3} : z_1, z_2 \rightarrow z_1^T w_3 z_2 $$

In these examples of zero-shot learning where a simple classical multi-class hinge loss was able to train classifiers [using precomputed output embedding for each class](https://arxiv.org/pdf/1409.8403.pdf), a [joint embedding loss](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Reed_Learning_Deep_Representations_CVPR_2016_paper.pdf) can train the two embeddings simultanously.

The joint embedding optimization can also be seen as **maximizing the log-likelihood for a binomial problem** where the output variable is the degree of similarity $$ s_{1,2} = [ y_1 == y_2 ] $$  and the input variable $$ X = (x_1,x_2) $$ the combined modalities :

$$ \log p_{w_1,w_2,w_3}( s_{1,2} | x_1, x_2 ) $$

where

$$ p( s_{1,2} | x_1, x_2 ) = \int p( z_1 | x_1) p( z_2 | x_2) p( s_{1,2} | z_1, z_2) dz_1 dz_2\\
\geq \max_{z_1,z_2} p(z_1|x_1) p(z_2|x_2) p(s_{1,2} | z_1, z_2 )  $$

Hence, maximizing

$$ \mathscr{L}(w) = \max_{z_1,z_2} \log p_{w_1}(z_1|x_1) + \log p_{w_2}(z_2|x_2) + \log p_{w_3}( s_{1,2} | z_1, z_2 ) $$

In [Zero shot learning via joint latent similarity Embedding](https://arxiv.org/pdf/1511.04512v3.pdf), Zhang *et al.* propose an algorithm that iteratively assigns to each example in the dataset an embedding value $$ (z_1, z_2) $$ that maximizes the objective function over all data, then optimizes $$ w = ( w_1,w_2,w_3 ) $$ for this assignment at a very good computational cost.

# Connectionist temporal classification loss

This loss function is designed for temporal classification, to have the underlying network concentrate its discriminative capacity and sharpen its classification around **spikes**.

This loss has been used in [audio classification](ftp://ftp.idsia.ch/pub/juergen/icml2006.pdf).

We consider sequences **x** of length T, depending of the sampling interval. The idea is to design a network that is able to predict the correct class as well as *blanks* during no class is predicted. The underlying network output is enhanced with a *blank* class, and the output dictionary becomes $$ L \cup \{ blank \} $$. At each time step or *audio frame*, the network will predict the class probability $$ y_k^t $$ and the probability of a *path* is given by

$$ p(\pi | x ) = \prod_{t=1}^T  y_k^t $$

Nevertheless not every path can be a label : in audio classification, if successive frames are too close, they will correspond to the same phoneme, that's why the first rule is to reduce successive identical predictions into 1 phoneme if there is no blank between them. Also blanks can be removed.

$$ \mathcal{B}(a--ab-) = \mathcal{B}(-aa-abb) = aab $$

and the probability of a label in the CTC loss is defined as the sum of the probabilities of all paths reducing to it :

$$ p(l | x ) = \sum_{\pi \in \mathcal{B}^{-1}(l)}  p(\pi | x ) $$

The CTC loss is simply the negative log likelihood of this probability for correct classification

$$ - \sum_{(x,l)\in \mathcal{S}} \ln p(l | x) $$

[The original paper](ftp://ftp.idsia.ch/pub/juergen/icml2006.pdf) gives the formulation to compute the derivative of the CTC loss.

**You're all set for choosing the right loss functions.**
