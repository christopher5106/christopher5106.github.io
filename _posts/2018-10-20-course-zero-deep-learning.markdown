---
layout: post
title:  "Course 0: deep learning in 5 days only!"
date:   2018-10-20 05:00:00
categories: deep learning
---

In this course, we'll first begin by checking where you are in data science, how far you have been, and if you have the basic concepts because main concepts in deep learning comes from datascience. Then, we'll go for a bit more in marchine learning, and introduction to deep learning, programming, coding few functions under a deep learning technologies, Pytorch, and I'll explain why Pytorch is a great technology. We'll compare it with other technologies, which is not so easy at the beginning. And then we'll go further and further with deep neural networks, to address different subjects.

I hope you'll get some feelings about deep learning you cannot get from reading else where.

# First concept: loss functions

When we fit a model, we use *loss functions*, or *cost functions*, or *objective function*. The main purpose of machine learning of machine learning is to be able to predict given data. Let's say predict the $$y$$ given some observations, let's say the $$x$$, through a function:

$$ f : x \rightarrow y $$

Usually, f is called a model and is parametrized, let's say by a list of parameters $$ \theta $$

$$ f = f_\theta $$

and the goal of machine learning is to find the best parameters.

In supervised learning we know the real value we want to predict $$\tilde{y}$$, for example the class of the object we want to predict.

We want to be able to measure the difference between what we are predicted with the model $$f$$ and what it should predict, what we call the **ground truth**. For that, there are some cost functions, depending on the problem you want to address.

$$L(y, \tilde{y}) $$

The two most important loss functions are:

- **Mean Squared Error (MSE)**, usually used for regression: $$ \sum_i (y_i - \tilde{y_i})^2 $$

- **Cross Entropy for probabilities**, in particular for classification where the model predicts the probability of the observed object x for each class

<center> $$ x \rightarrow \{p_c\}_c $$ with $$ \sum_c p_c = 1 $$ </center>

Transforming any function output into a probability that sums to 1 is usually performed thanks to a softmax function

$$ x \xrightarrow{f} o = f(x) = \{o_c\}_c \xrightarrow{softmax} \{p_c\}_c $$

where the softmax normalization function is defined by:

$$ \text{Softmax}(o) = \Big\{ \frac{ e^{-o_i} }{ \sum_c e^{-o_c}}  \Big\}_i $$

Note that for the softmax to predict probability for c classes, it requires the output $$ o = f(x) $$ to be c-dimensional.

For example, in image classification, X being the image of a cat, we want this output $$\{p_c\}_c $$ to fit the real class probability $$\{\tilde{p}_c\}_c $$, where $$ p_\hat{c} = 1 $$ for the real object class $$ \hat{c} $$ "cat" and $$ p_c = 0$$ for all other classes $$ c \neq \hat{c} $$:

<img src="{{ site.url }}/img/deeplearningcourse/DL1.png">

Coming from the theory of information, cross-entropy is a distance measure between two probabilities defined by :

$$ \text{CrossEntropy}(p, \tilde{p}) = - \sum_c \tilde{p}_c \log(p_c) = - \log(p_\hat{c})$$

we want to be the highest possible (maximization).

To discover many more loss functions, have a look at my [full article about loss functions](http://christopher5106.github.io/deep/learning/2016/09/16/about-loss-functions-multinomial-logistic-logarithm-cross-entropy-square-errors-euclidian-absolute-frobenius-hinge.html).

Note that **a loss function always outputs a scalar value**. This scalar value is a measure of fit of the model with the real value.

In **conclusion** of this section, the goal of machine learning is to have a function fit with the real world; and to have this function fit well, we use a loss function to measure how to reduce this distance.

# Second concept: the Gradient Descent

To minimize the cost function, the most used technique is the Gradient Descent.

It consists in following the gradient to descend to the minima:

<img src="{{ site.url }}/img/deeplearningcourse/DL2.png">


It is an iterative process in which the update rule simply consists in:

$$ \theta_{t+1} = \theta_{t} - \lambda \nabla_\theta (L \circ f_\theta) $$

where L is our cost $$ L(f_\theta(x), \tilde{y}) $$

<img src="{{ site.url }}/img/deeplearningcourse/DL3.png">

An **update rule** is how to update the parameters of the model to minimize the cost function.

$$\lambda $$ is the learning rate and has to be set carefully: Effect of various learning rates on convergence (Img Credit: cs231n)

<img src="{{ site.url }}/img/deeplearningcourse/DL4.png">


This simple method is named SGD, after *Stochastic Gradient Descent*. There are many improvements around this simple rule: ADAM, ADADELTA, RMS Prop, ... All of them are using the first order only. Some are adaptive, such as ADAM or ADADELTA, where the learning rate is adapted to each parameter automatically.

**Conclusion**: When we have the loss function, the goal is to minimize it. For this, we have a very simple update rule which is the gradient descent.


# Backpropagation

Computing the gradients to use in the SGD update rule is known as *backpropagation*.

The reason for this name is the *chaining rule* in computing gradients of function compositions.

In fact, models are usually composed of multiple functions:


$$ L = \text{CrossEntropy} \circ \text{Softmax} \circ f_\theta (x)$$


$$ = \text{CrossEntropy} \circ \text{Softmax} \circ \text{Dense}_{\theta_2}^2 \circ \text{ReLu} \circ \text{Dense}_{\theta_1}^1  (x) $$

where the composition means

$$ L = \text{CrossEntropy} (\text{Softmax} (\text{Dense}_\theta^2 ( \text{ReLu} ( \text{Dense}_\theta^1  (x) ) ) ) ) $$

Here, for example, I have two dense layers and two activations (one ReLu and one Softmax). There are two parameters

$$ \theta = [ \theta_1, \theta_2 ] $$

The chaining rules for gradient computation says:

$$ \frac{ \partial }{ \partial x_i} (f \circ g )_k = \sum_c \frac{\partial f_k}{\partial g_c} \cdot \frac{\partial g_c}{\partial x_i} $$

which can be rewritten as a simple matrix multiplication

$$ \nabla (f \circ g) = \nabla f \times \nabla g $$


What does that mean for deep learning and gradient descent ? In fact, for each layer, we want to compute

$$ \nabla_{\theta_{\text{Layer}}} \text{Layer} $$

given layer's input to update its parameters $$ \theta_{\text{Layer}} $$.

So, for the layer $$ \text{Dense}^2 $$,

$$ x \xrightarrow{ \text{Dense}^1 }  \xrightarrow{ \text{ReLu} } y  \xrightarrow{ \text{Dense}^2 }  \xrightarrow{ \text{Softmax}}  \xrightarrow{ \text{CrossEntropy} } L $$

the gradient is given by

$$ \nabla_{\theta_2} L = \nabla_{\theta_2} ( \text{CrossEntropy} \circ \text{Softmax} \circ \text{Dense}^2 ) $$

$$ = \nabla_{CE inputs}  \text{CrossEntropy} \times \nabla_{Softmax inputs} \text{Softmax} \times \nabla_{\theta_2} \text{Dense}^2 $$

<img src="{{ site.url }}/img/deeplearningcourse/DL8.png">

Integrating into the full model  

**Conclusion**:

# Cross Entropy with Softmax

Let's come back to the global view: an observation X, a model f depending on parameters $$\theta $$, a softmax to normalize the model's output, and last, our cross entropy outputting a final scalar, measuring the distance between the prediction and the expected value:

<img src="{{ site.url }}/img/deeplearningcourse/DL5.png">

The cross entropy is working very well with the softmax function and is usually implemented as one layer, for numerical efficiency.

Let us see why and study the combination of softmax and cross entropy

$$ o = f(x) = \{o_c\}_c \xrightarrow{ \text{Softmax}}  \xrightarrow{ \text{CrossEntropy} } L $$

Mathematically,

$$ L =  - \log(p_\hat{c}) = - \log \Big(  \frac{ e^{-o_\hat{c}} }{ \sum_c e^{-o_i}}  \Big) $$

$$ = - \log e^{-o_\hat{c}} +  \log \sum_c e^{-o_i}  $$

$$ = o_\hat{c} +  \log \sum_c e^{-o_i} $$

Let's take the derivative,

$$ \frac{\partial L}{\partial o_c} =  \delta_{c,\hat{c}} - \frac{ e^{-o_i} }{\sum_c e^{-o_i}}  $$

$$ = \delta_{c,\hat{c}} - p_c $$

which is very easy to compute and can simply be rewritten:

$$ \nabla L = \tilde{p} - p $$

**Conclusion**: it is easier to backprogate gradients computed on Softmax+CrossEntropy together rather than backpropagate separately each : the derivative of the Softmax+CrossEntropy with respect to the output of the model for the right class, let's say the "cat" class, will be 1 - 0.8 = 0.2, if the model has predicted a probability of 0.8 for this class, encouraging it to increase this value ; the derivative of the Softmax+CrossEntropy with respect to an output for a different class will be -0.4 is it has predicted a probability of 0.4, encouraging the model to decrease this value.  


# Example with a Dense layer

Let's take, as model, a very simple one with only one Dense layer with 2 filters in one dimension and an input of dimension 2:

<img src="{{ site.url }}/img/deeplearningcourse/DL7.png">

Such a layer produces a vector of 2 scalars:

$$ f_\theta : \{x_j\} \rightarrow \Big\{ o_i = \sum_j \theta_{i,j} x_j \Big\}_i$$

Please keep in mind that it is not possible to descend the gradient directly on this output because it is composed of two scalars. We need a loss function, that returns a scalar, and tells us how to combine these two outputs. For example, the Softmax+CrossEntropy we have seen previously:

$$ L = \text{CrossEntropy}(\text{Softmax}(o)) $$

<img src="{{ site.url }}/img/deeplearningcourse/DL6.png">

Now we can retropropagate the gradients. Since we are in the case of 2 outputs, we call this problem a binary classification problem, let's say: cats and dogs.

Let us compute the derivatives of the dense layer:

$$ \frac{\partial o_i}{\partial \theta_{k,j}} = \begin{cases}
  x_j, & \text{if } k = i, \\
  0, & \text{otherwise}.
\end{cases} $$

so

$$ \frac{\partial}{\partial \theta_{k,j}}  ( L \circ f_\theta )=  \sum_c \frac{\partial L}{\partial o_c}  \cdot \frac{\partial o_c}{\partial \theta_{k,j}}  = ( \delta_{ k, \hat{c}} - v_k) \cdot x_j  $$

**Well done!**

Now let's go to next course: [Course 1: programming deep learning!](http://christopher5106.github.io/deep/learning/2018/10/20/course-one-programming-deep-learning.html)
