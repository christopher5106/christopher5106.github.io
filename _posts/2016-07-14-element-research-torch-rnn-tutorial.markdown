---
layout: post
title:  "Element-Research' Torch RNN Tutorial"
date:   2016-07-14 17:00:51
categories: deep learning
---

It is quite hard to understand the `torch.rnn` package since it begins the description with abstract classes.

Let's begin with simple examples and put things back into order to simplify comprehension for beginners.


# Install

To use Torch with NVIDIA library CUDA, I would advise to copy the CUDA files to a new directory `/usr/local/cuda-8-cudnn-5` in which you install the last CUDNN since Torch prefers a CUDNN version above 5. The [install guide](http://torch.ch/docs/getting-started.html) will step up Torch. In your `~/.bashrc` :

```bash
export LD_LIBRARY_PATH=/usr/local/cuda-8.0-cudnn-5/lib64:/usr/local/cuda-8.0-cudnn-4/lib64:/usr/local/lib/
. /home/christopher/torch/install/bin/torch-activate
```

Install the `rnn` package and the [dependencies](https://github.com/Element-Research/rnn)

    luarocks install torch
    luarocks install nn
    luarocks install dpnn
    luarocks install torchx
    luarocks install cutorch
    luarocks install cunn
    luarocks install cunnx
    luarocks install rnn

Launch a Torch shell with `luajit` command.

# Build a simple RNN

Let's build a simple RNN like this one :

![simple RNN]({{ site.url }}/img/rnn.png)

with an hidden state of size 7 to predict the new word in a dictionary of 10 words.

**Compute the hidden state at each time step**

$$ h_t = \sigma(W_{hh} h_{t−1} + W_{xh} X_t) $$

The $$ W_{xh} $$ is a kind of word embedding where the multiplication with the word index produces the vector representing the word in the embedding space (in this case it's a 7-dim space).

This equation is implemened by the Reccurent layer :

```lua
require 'rnn'

r = nn.Recurrent(
    7,
    nn.LookupTable(10, 7),
    nn.Linear(7, 7),
    nn.Sigmoid(),
    8
  )
=r
```

The `nn.Recurrent` takes 6 arguments :

- 7, the size of the hidden state

- `nn.LookupTable(10, 7)` giving the $ W_{xh} X_t $$

- `nn.Linear(7, 7)` describing the $W_{hh} h_{t−1}$

- `nn.Sigmoid()` if the activation function

- 8 (the `rho` parameter) is the maximum number of steps to backpropagate through time (BPPT). It can be initialiazed to 9999 or the size of the sequence.

So far, we have built this part :

![simple RNN]({{ site.url }}/img/rnn_hidden.png)

This module inherits from the **AbstractRecurrent** interface (abstract class).

There is a new way to create the same net, thanks to the new and more general `Recurrence` module :

```lua
rm = nn.Sequential()
   :add(nn.ParallelTable()
      :add(nn.LookupTable(10, 7))
      :add(nn.Linear(7, 7)))
   :add(nn.CAddTable())
   :add(nn.Sigmoid())

r = nn.Recurrence(rm, 7, 1)   
```
where

- rm has to be a module that computes $h_t$ given an input table $ X_t, h_{t-1} $

- 7 is the hidden state size

- 1 is the input dimension


**Compute the output at a time step**

Now, let's add the output to our previous net.

Output is computed thanks to

$$ o_t = W_{ho} h_t $$

and converted to a probability with the Softmax function.

```lua
rr = nn.Sequential()
   :add(r)
   :add(nn.Linear(7, 10))
   :add(nn.LogSoftMax())
=rr
```

So far, we have built this part :

![simple RNN]({{ site.url }}/img/rnn_step.png)

This module does not inherit anymore from the **AbstractRecurrent** interface (abstract class).

**Make it a recurrent module**

The previous module is not recurrent. It is still taking one input at time.

`Recurrent`, `LSTM`, `GRU`, ... are recurrent modules. They can be applied to sequences.

Since we added non-recurrent modules, we have to transform the net back to a recurrent module, with the `Recursor` function :

```lua
rnn = nn.Recursor(rr, 8)
=rnn
```

This will clone the non-recurrent submodules, each clone sharing the same parameters and gradients for the parameters.

Now we have

![simple RNN]({{ site.url }}/img/rnn.png)

**Apply the net to each element of a sequence step by step**

Let's apply our recurring net to a sequence of 7 words given by an input `torch.LongTensor` and compute the error with the expected target `torch.LongTensor`.

```lua
outputs, err = {}, 0
criterion = nn.ClassNLLCriterion()
for step=1,7 do
   outputs[step] = rnn:forward(inputs[step])
   err = err + criterion:forward(outputs[step], targets[step])
end
```

**Train the net step by step through the sequence**

Let's retropropagate the error through time, going in the reverse order of the forwards:

```lua
gradOutputs, gradInputs = {}, {}
for step=7,1,-1 do
  gradOutputs[step] = criterion:backward(outputs[step], targets[step])
  gradInputs[step] = rnn:backward(inputs[step], gradOutputs[step])
end
```
and update the parameters

```lua
rnn:updateParameters(0.1) -- learning rate
```

**Reset the net**

To reset the hidden state after a forward call (training and evaluation) :

```lua
rnn:forget()
```

To reset the accumulated gradients for the parameters :

```lua
rnn:zeroGradParameters()
```

# Apply a net to a sequence in one step thanks to sequencer module

The `Sequencer` module enables to transform a net to apply it directly to the full sequence:

```lua
rnn = nn.Sequencer(rr)
criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())
```

which is the same as `rnn = nn.Sequencer(nn.Recursor(rr))` because the sequence does it internally.

Then, there is only one global forward and backward pass as if it were a feedforward net :

```lua
outputs = rnn:forward(inputs)
err = criterion:forward(outputs, targets)
gradOutputs = criterion:backward(outputs, targets)
gradInputs = rnn:backward(inputs, gradOutputs)
rnn:updateParameters(0.1)
```

Since the Sequencer takes care of calling the `forget` method, just reset the gradient parameters for the next training step :

```lua
rnn:zeroGradParameters()
```

# Prebuilt RNN and Sequencers

There exists a bunch of RNN :

- `nn.LSTM` and `nn.FastLSTM` (a faster version)
- `nn.GRU`

as well as sequencers :
- `nn.seqLSTM` (a faster version than `nn.Sequencer(nn.FastLSTM(inputsize, outputsize))`)
- `nn.seqGRU`
- `nn.BiSequencer` to transform a RNN into a bidirectionnal RNN
- `nn.SeqBRNN` a bidirectionnal LSTM
- `nn.Repeater` a simple repeat layer (which is not a RNN)
- `nn.RecurrentAttention`

# Helper functions

- `nn.SeqReverseSequence` to reverse a sequence order
- `n.SequencerCriterion`
- `nn.RepeaterCriterion`

**Well done!**
