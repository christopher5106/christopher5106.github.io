---
layout: post
title:  "Course 0: deep learning!"
date:   2018-10-20 00:00:00
categories: deep learning
---

In this course, we'll first begin by checking where you are in data science, how far you have been, and if you have the basic concepts because main concepts in deep learning comes from datascience. Then, we'll go for a bit more in marchine learning, and introduction to deep learning, programming, coding few functions under a deep learning technologies, Pytorch, and I'll explain why Pytorch is a great technology. We'll compare it with other technologies, which is not so easy at the beginning. And then we'll go further and further with deep neural networks, to address different subjects.

I hope you'll get some feelings about deep learning you cannot get from reading else where.

# First concept: loss functions

When we fit a model, we use *loss functions*, or *cost functions*. The main purpose of machine learning of machine learning is to be able to predict given data. Let's say predict the $$y$$ given some observations, let's say the $$x$$, through a function:

$$ f : x\arrow y $$

In supervised learning we know the real value we want to predict $$\tild{y}$$, for example the class of the object we want to predict.

We want to be able to measure the difference between what we are predicted with the model $$f$$ and what it should predict, what we call the **ground truth**. For that, there are some cost functions, depending on the problem you want to address.

$$d(y, \tild{y}) $$
