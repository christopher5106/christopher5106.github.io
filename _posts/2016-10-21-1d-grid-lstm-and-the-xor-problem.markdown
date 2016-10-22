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

It is a much more complicated problem to train a feed forward network with the traditional gradient descent on the full bit string as input. The 1-Grid LSTM is a feed forward network that solves this problem very nicely, demonstrating the power of the grid.

# The story

Grid LSTM have been invented in continuity of stacked LSTM and multi-dimensional LSTM. They present a more general concept, much precise than stacked LSTM and more stable than multi-dimensional LSTM.

Since it is complicated to visualize a N-Grid LSTM let's consider the 2-Grid-LSTM.

The 2-Grid-LSTM is composed of 2 LSTM running on 2 different dimensions (x,y), given by the following equation :

$$ h_2^{x,y}, m_2^{x,y} = \text{LSTM}^2 (h_1^{x,y-1}, h_2^{x,y-1}, m_2^{x,y-1}) $$

$$ h_1^{x,y}, m_1^{x,y} = \text{LSTM}^1 (h_1^{x-1,y}, h_2^{x,y}, m_1^{x-1,y}) $$

To get a good idea of what it means, you can consider a 2-Grid-LSTM as a virtual cell traveling on a 2 dimensional grid :

![The run of a 2-GRID LSTM]({{ site.url }}/img/grid-2d.png)

LSTM 1 uses the hidden state value of the LSTM 2, in this case $$ h_2^{x,y} $$, instead of its previous value. This helps the network learn faster correlation between the two LSTM. There is always a notion of **priority** and **order** in the computation of the LSTM. Parallelisation of the computations require to take into consideration this order.

For each cell in the grid, there can be an input and/or an output, as well as no input and/or no output.

In the case of 1D sequences (times series, textes, ...) for which 2-GRID LSTM perform better than stacked LSTM, input is given for the bottom row, as shown with the input sentence "I love deep learning" in the above figure.

In case of text classification, output will be given for the last cell top right (as shown with "Good" output label in the above figure).

The first dimension is usually the *time* (x), the second dimension the *depth* (y).

Here is the mecanism and the connections inside the cell :

![]({{ site.url }}/img/grid-2d-lstm.png)


# 1-Grid LSTM, a special case

The 1-Grid LSTM is a special case since the LSTM is used along the depth dimension to create a feedforward classifier instead of a recurrent network. It looks as follow :

![]({{ site.url }}/img/grid-1d.png)

It looks very closely to a LSTM, but with the following differences :

- there is no input at each step in the recurrence of the LSTM, and the recurrence is used along the depth dimension as a gated mecanism (such as in a Highway networks)

- the input is fed into the hidden state and cell state thanks to a linear projection

The XOR problem shows the power of such network as classifiers. Here are two implementations :

- [Theano 1-Grid LSTM](https://github.com/christopher5106/grid-1D-LSTM-theano)

- [Torch 1-Grid LSTM](https://github.com/christopher5106/grid-1D-LSTM-torch)


# Generalization

The N-Grid LSTM is a improvement of the multi-dimensional LSTM.

The option to untie the weigths in the depth direction is considered also during evaluation of models.

Two options are possible :

- the LSTM 1 at different level y can have different weigths. It is usually the case in many implementations.

- the LSTM 2 can be transformed into a feedforward model, where weights are not shared between the different level y.

Last, it is also possible to remove the cell in the LSTM 2 and replace the LSTM mecanism by a non-linearity, to come back to the stacked LSTM architecture.

In this sense, N-Grid-LSTM with the option of **untying the weights** and modifying the mecanism is a generalization of stacked LSTM. 

**Nice!**
