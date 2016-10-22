---
layout: post
title:  "1-grid LSTM and the XOR problem"
date:   2016-10-21 17:00:51
categories: deep learning
---

A short post about a very nice classifier, the 1-Grid-LSTM, and its application to the XOR problem, as proposed by [Nal Kalchbrenner, Ivo Danihelka, Alex Graves in their paper](https://arxiv.org/abs/1507.01526).

The XOR problem consists in predicting the parity of a string of bits, 0 and 1. If the sum of the bits is odd, the target is the 0. If the sum of the bits is even, the target is 1.

The problem is pathological in the sense a simple bit in the input sequence can change the target to its contrary. During training, accuracy is not better than random for a while, although a kind of engagement can be seen in a wavy increase to 100%.

Building a network manually, or building a recurrent network with the bits as input sequence to the RNN, are easy.

It is a much more complicated problem to train a feed forward network with the traditional gradient descent on the full bit string as input. The 1-Grid LSTM solves this problem very nicely, demonstrating the power of the grid.

# The story

Grid LSTM have been invented in continuity of stacked LSTM and multi-dimensional LSTM. They present a more general concept, much precise than stacked LSTM and more stable than multi-dimensional LSTM.

Consider a 2-Grid-LSTM (2 dimensions) as a virtual cell running on a 2D grid  instead of a sequence.

![The run of a 2-GRID LSTM]({{ site.url }}/img/grid-2d.png)

For each cell in the grid, there can be an input and/or an output, as well as no input and/or no output. In the case of 1D sequences (times series, textes, ...) for which GRID LSTM perform better, input can be given for the bottom row only as shown with the input sentence "I love deep learning". In case of text classification for example, output will be given for the last cell top right (as shown with "Good" output label).

The first dimension is usually the *time*, the second dimension the *depth*.

The equation is given by :

$$ h_2', m_2' = \text{LSTM} (h_1, h_2', m_2, W_2) $$

$$ h_1', m_1' = \text{LSTM} (h_1, h_2, m_1, W_1) $$


In this example, the depth dimension is prioritary on the time dimension : we first compute the depth dimension, then the time dimension second, since the LSTM of the time dimension will rely on the output value of the depth dimension. Parallelisation of the computations cannot be performed randomly and require to follow this order.

![]({{ site.url }}/img/grid-2d-lstm.png)

We usually feed the input into the hidden state of the first row, with a linear state. Let's see with 1D Grid LSTM.

# 1-Grid LSTM

The 1-Grid LSTM looks as follow :

![]({{ site.url }}/img/grid-1d.png)

It looks very closely to a LSTM, but with the following differences :

- there is no input at each step in the recurrence of the LSTM

- the input is fed into the hidden state and cell state thanks to a linear projection

The XOR problem shows the power of such network as classifiers. Here are two implementations :

- [Theano 1-Grid LSTM](https://github.com/christopher5106/grid-1D-LSTM-theano)

- [Torch 1-Grid LSTM](https://github.com/christopher5106/grid-1D-LSTM-torch)

# Generalization

The N-Grid LSTM is a improvement of the multi-dimensional LSTM.

The option to untie the weigths in the depth direction is considered also during evaluation of models.

In this case, if you also remove the cell in the depth LSTM and replace the LSTM by a non-linearity, you come back to the stacked LSTM.
