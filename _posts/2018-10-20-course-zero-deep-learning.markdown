---
layout: post
title:  "Course 0: deep learning!"
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

$$ softmax(o) = \Big\{ \frac{ e^{-o_i} }{ \sum_c e^{-o_i}}  \Big\}_i $$

Note that for the softmax to predict probability for c classes, it requires the output $$ o = f(x) $$ to be c-dimensional.

For example, in image classification, X being the image of a cat, we want this output $$\{p_c\}_c $$ to fit the real class probability $$\{\tilde{p}_c\}_c $$, where $$ p_\hat{c} = 1 $$ for the real object class $$ \hat{c} $$ "cat" and $$ p_c = 0$$ for all other classes $$ c \neq \hat{c} $$:

<img src="{{ site.url }}/img/deeplearningcourse/DL1.png">

Coming from the theory of information, cross-entropy is a distance measure between two probabilities defined by :

$$ crossentropy(p, \tilde{p}) = \sum_c \tilde{p}_c \log(p_c) = \log(p_\hat{c})$$

we want to be the highest possible (maximisation).

In conclusion of this section, the goal of machine learning is to have a function fit with the real world; and to have this function fit well, we use a loss function.

# Second concept: the Gradient Descent

To minimize the cost function, the most used technique is the Gradient Descent.

It consists in following the gradient to descend to the minima:

<img src="{{ site.url }}/img/deeplearningcourse/DL2.png">


It is an iterative process in which the update rule simply consists in:

$$ \theta_{t+1} = \theta_{t} - \lambda_\theta L $$

where L is our cost $$ \L(f_\theta(x), \tilde{y}) $$

<img src="{{ site.url }}/img/deeplearningcourse/DL3.png">

An **update rule** is how to update the parameters of the model to minimize the cost function.

$$\lambda $$ is the learning rate and has to be set carefully:

<img src="{{ site.url }}/img/deeplearningcourse/DL4.png">
Effect of various learning rates on convergence (Img Credit: cs231n)

This simple method is named SGD, after *Stochastic Gradient Descent*. There are many improvements around this simple rule: ADAM, ADADELTA, RMS Prop, ... All of them are using the first order only. Some are adaptive, such as ADAM or ADADELTA, where the learning rate is adapted to each parameter automatically.

Conclusion: When we have the loss function, the goal is to minimize it. For this, we have a very simple update rule which is the gradient descent.


# Cross Entropy with Softmax

Let's come back to the global view: an observation X, a model f depending on parameters $$\theta $$, a softmax to normalize the model's output, and last, our cross entropy outputting a final scalar, measuring the distance between the prediction and the expected value: 

<img src="{{ site.url }}/img/deeplearningcourse/DL5.png">

The cross entropy is working very well with the softmax function and is usually implemented as one layer, for numerical efficiency.

Let us see why:

$$ \log(p_\hat{c}) = \log \Big(  \frac{ e^{-o_\hat{c}} }{ \sum_c e^{-o_i}}  \Big) $$

$$ = \log e^{-o_\hat{c}} -  \log \sum_c e^{-o_i}  $$

Let's take the derivative of
