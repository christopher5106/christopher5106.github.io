---
layout: post
title:  "Course 1: programming deep learning in 5 days only!"
date:   2018-10-20 06:00:00
categories: deep learning
---

You might first check [Course 0: deep learning!](http://christopher5106.github.io/deep/learning/2018/10/20/course-zero-deep-learning.html) if you have not read it.

In this deep learning course, we'll use Pytorch as deep learning framework, the most modern technology in the area.

# Your programming environment

Deep learning demands heavy computations so all deep learning libraries offer the possibility of parallel computing on GPU rather CPU, and distributed computed on multiple GPUs or instances.

The use of specific hardwares such as GPUs requires to install an up-to-date driver in the operating system first.

While OpenCl (not to confuse with OpenGL or OpenCV) is an open standard for GPU programming, the most used GPU library is CUDA, a private library by NVIDIA, to be used on NVIDIA GPUs only.

CUDNN is a second library coming with CUDA providing you with more optimized operators.

Once installed on your system, these libraries will be called by more high level deep learning frameworks, such as Caffe, Tensorflow, MXNet, CNTK, Torch or Pytorch.

The command `nvidia-smi` enables you to check the status of your GPUs, as with `top` or `ps` commands.

Most recent GPU architectures are Pascal and Volta architectures. The more memory the GPU has, the better. Operations are usually performed with single precision `float16` rather than double precision `float32`, and on new Volta architectures offer Tensor cores specialized with half precision operations.

One of the main difficulties come from the fact that different deep learning frameworks are not available and tested on all CUDA versions, CUDNN versions, and even OS. CUDA versions are not available for all driver versions and OS as well. A good solution to adopt is to use Docker files, which limits the choice of the driver version in the operating system: the compliant CUDA and CUDNN versions as well as the deep learning frameworks can be installed inside the Docker container.


# The batch

When applying the update rule, the best is to compute the gradients on the whole dataset but it is too costly. Usually we use a batch of training examples, it is a trade-off that performs better than a single example and is not too long to compute.

The learning rate needs to be adjusted depending on the batch size. The bigger the batch is, the bigger the learning rate can be.

So, most deep learning programms and frameworks consider the first dimension in your data as the batch size. All other dimensions are the data dimensionality. For an image, it is `BxHxWxC`, written as a shape `(B, H, W, C)`. After a few layers, the shape of the data will change to `(b, h, w, c)` : the batch size remains, the number of channels usually increases $$ c \geq C $$ and the feature map decreases $$ h \leq H, w \leq W$$ for top layers' outputs' shapes.

This format is very common and called *channel last*. Some deep learning frameworks work with *channel first*, such as CNTK, or enables to change the format as in Keras, to `(B, C, W, H)`.


<img src="{{ site.url }}/img/deeplearningcourse/DL14.png">

To distribute the training on multiple GPU or instances, the easiest way is to split along the batch dimension, which we call *data parallellism*, and dispatch the different splits to their respective instance. The parameter update step requires to synchronize more or less the gradient computations. NVIDIA provides fast multi-gpu collectives in its library NCCL, and fast connections between GPUs with NVLINK2.0.


# Training curves and metrics

As we have seen on [Course 0](http://christopher5106.github.io/deep/learning/2018/10/20/course-zero-deep-learning.html), we use a *cost function* to fit the model to the goal.

So, during training of a model, we usually plot the **training loss**, and if there is no bug, it is not surprising to see it decreasing as the number of training steps or iterations grows.

<img src="{{ site.url }}/img/deeplearningcourse/DL16.png">

Nevertheless, we usually keep 2 to 10 percent of the training set aside from the training process, which we call the **validation dataset** and compute the loss on this set as well. Depending if the model has enough capacity or not, the **validation loss** might increase after a certain step: we call this situation **overfitting**, where the model has too much learned the training dataset, but does not generalize on unseen examples. To avoid this situation to happen, we monitor the validation metrics as well to decide when to stop the training process, after which the model will perform less.

On top of the loss, it is possible to monitor other metrics, such as for example the accuracy. Metrics might not be differentiable, and minimizing the loss might not minimize the metrics. In image classification, a very classical one is the accuracy, that is the ratio of correctly classified examples in the dataset.

We also usually compute the precision/recall curve: precision defines the number of true positive in the examples predicted as positive by the model (true positives + false positives) while the recall is the number of true positives of the total number of positives (true positives + false negatives). While for some applications, such as document retrieval, we prefer to have higher recall, for some other applications, such as automatic document classification, we prefer to have a high precision for automatically classified documents, and leave ambiguities to a human operators. The area under the precision/recall curve (AUC), gives a good estimate of the discrimination quality of our model.

<img src="{{ site.url }}/img/deeplearningcourse/DL11.png">


# A library for deep learning

A deep learning library offers the following characteristics :

1. It works very well with Numpy arrays. Arrays are called **Tensors**. Moreover, operations on Tensors follow lot's of Numpy conventions.

2. It provides abstract classes, such as Tensors, for parallel computing on GPU rather CPU, and distributed computing on multiple GPUs or instances, since Deep learning demands huge computations.

3. Operators have a 'backward' implementation, computing the gradients for you, with respect to the inputs or parameters.

Let's load Pytorch Python module into a Python shell, as well as Numpy library, check the Pytorch version is correct and the Cuda library is correctly installed:

```python
import torch
import numpy as np
print(torch.__version__) # 0.4.1
print(torch.cuda.is_available()) # True
```

#### 1. Numpy compatibility

You can easily check the following commands in Pytorch and Numpy:

| ####### Command ####### | ####### Numpy ####### | ####### Pytorch ####### |
|---|---|---|
| Array conversion | np.array([[1,2]]) | torch.Tensor([[1,2]]) |
| 5x3 matrix, uninitialized  |  x = np.empty((5,3)) | x = torch.Tensor(5,3)  |
| initialized with ones | x = np.ones((5,3)) | x = torch.ones(5,3) |
| initialized with zeros | x = np.zeros((5,3)) | x = torch.zeros(5,3) |
| uniformly randomly initialized matrix  |  x = np.random.rand(5,3) | x = torch.rand(5, 3)  |
| normal randomly initialized matrix  |  x = np.random.randn(5,3) | x = torch.randn(5, 3)  |
| Shape/size  |  x.shape | x.size  |
| Addition | +/np.add | +/torch.add |
| In-place addition | x+= | x.add_() |
| First column | x[:, 1] | x[:, 1] |
| Matrix multiplication | .matmul() | .mm() |
| Matrix-Vector multiplication | - | .mv() |
| Reshape | .reshape(shape) | .view(size) |
| Concatenate | np.concatenate([]) | torch.cat([]) |
| Add a dimension | np.expand_dims(, axis) | .unsqueeze(axis) |
| Range of values | np.arange() | torch.arange() |




You can link Numpy array and Torch Tensor, either with

```python
numpy_array = np.ones(5)
torch_array = torch.from_numpy(numpy_array)
```

or

```python
torch_array = torch.ones(5)
numpy_array = torch_array.numpy()
```

which will keep the pointers to the same values:

```python
numpy_array+= 1
print(torch_array) # tensor([2., 2., 2., 2., 2.])
torch_array.add_(1)
print(numpy_array) # [3. 3. 3. 3. 3.]
```

#### 2. GPU computing

It is possible to transfer tensors values between devices, ie RAM memory and each GPUs' memory:

```python
a = torch.ones(5,) # tensor([1., 1., 1., 1., 1.])
b = torch.ones(5,)
a_ = a.cuda() # tensor([1., 1., 1., 1., 1.], device='cuda:0')
b_ = b.cuda()
x = a_ + b_ # tensor([2., 2., 2., 2., 2.], device='cuda:0')
y = x + 1 # tensor([3., 3., 3., 3., 3.], device='cuda:0')
z = y.cuda(1) # tensor([3., 3., 3., 3., 3.], device='cuda:1')
t = z.cpu() # tensor([3., 3., 3., 3., 3.])
```

but keep in mind that synchronization is lost (contrary to Numpy Arrays and Torch Tensors):

```python
a.add_(1)
print(a_) # tensor([1., 1., 1., 1., 1.], device='cuda:0')
```

Tensors behave as classical programming non-reference variables and their content is copied from device to the other. This way, you decide when to transfer the data.

Contrary to other frameworks, Pytorch does not require to build a graph of operators and execute the graph on a device. Pytorch programming is as normal Python programming.

#### 3. Automatic differentiation

To compute the gradient automatically, you need to wrap the tensors in Variable objects:

```python
x = torch.autograd.Variable(torch.ones(1)+1, requires_grad=True)
print(x.data)
#tensor([2.])
print(x.grad)
# None
print(x.grad_fn)
# None
```

The Variable contains the original data in the `data` attribute. The API for Tensors is also available for Variables.

When you add an operation such as for example the square operator, the newly created Variable is populated with the result in the `data` attribute as well as an history `grad_fn` function :

```python
y = x ** 2
print(y.data)
# tensor([4.])
print(y.grad)
# None
print(y.grad_fn)
# <PowBackward0 object at 0x7ffaf1ae0160>
```

The `grad_fn` Function contains the history or the link to x, which enables to compute the derivative thanks to the Variable's `backward()` method:

```python
y.backward()
print(x.data)
# tensor([2.])
print(x.grad)
# tensor([4.])
```

The gradient of the cost y with respect to the input x is placed in `x.grad` and since $$ \frac{\partial x^2}{\partial x} = 2 x $$, its value is 4 at x=2.

Calling `y.backward()` a second time will lead to a RunTime Error. In order to accumulate the gradients into `x.grad`, you need to set `retain_graph=True` during the first backward call.

Let's confirm this in a  case where the input is multi-dimensional:

```python
x = torch.autograd.Variable(torch.ones(2), requires_grad=True)
y = x.sum()
y.backward(retain_graph=True)
print(x.grad)
# tensor([1., 1.])
y.backward()
print(x.grad)
# tensor([2., 2.])
y.backward()
print(x.grad)
# tensor([3., 3.])
```

Since $$ \frac{\partial}{\partial x_1} (x_1 + x_2) = 1 $$ and $$ \frac{\partial}{\partial x_2} (x_1 + x_2) = 1 $$, the `x.grad` tensor is populated with ones.

Applying the `backward()` method multiple times accumulates the gradients.

It is also possible to apply the `backward()` method on something else than a cost (scalar), for example on a layer or operation with a multi-dimensional output, as in the middle of a neural network, but in this case, you need to provide as argument to the `backward()` method $$ \Big( \nabla_{I_{t+1}} \text{cost} \Big)$$, the gradient of the cost with respect to the output of the current operator/layer (which is written here as the input of the operator/layer above), which will be multiplied by $$ \Big( \nabla_{\theta_t} L_t \Big) $$, the gradient of the current operator/layer's output with respect to its parameters, in order to produce the gradient of the cost with respect to its parameters:

$$ \nabla_{\theta_t} \text{cost} =  \nabla_{\theta_t} \Big[ ( \text{cost} \circ S \circ ... \circ L_{t+1}  ) \circ L_t \Big]   = \Big( \nabla_{I_{t+1}} \text{cost} \Big) \times \nabla_{\theta_t} L_t  $$

as given by the chaining rule seen in [Course 0](http://christopher5106.github.io/deep/learning/2018/10/20/course-zero-deep-learning.html).

```python
x = torch.autograd.Variable(torch.ones(2), requires_grad=True)
y = x ** 2
print(y.data)
# tensor([1., 1.])
y.backward(torch.tensor([-1., 1.]))
print(x.grad)
# tensor([-2.,  2.])
```

The gradient of the final cost with respect to the output of the current operator/layer indicates how to combine the derivatives of different output values in the current layer in the production of a derivative with respect to each parameter.

As Pytorch does not require to introduce complex graph operators as in other technologies (switches, comparisons, dependency controls, scans/loops... ), it enables you to program as normally, and gradients are well propagated through your Python code:

```python
x = torch.autograd.Variable(torch.ones(1) +1, requires_grad=True)
y = x
while y < 10:
  y = y**2
print(y.data)
# tensor([16.])
y.backward()
print(x.grad)
# tensor([32.])
```

which is fantastic. In this case, $$ y\vert_{x=2} = ( x^2 )^2 = x^4  $$ and  $$ \frac{\partial y}{\partial x} \big\vert_{x=2} = 4 x^3 = 32 $$.

Note that gradients are computed by retropropagate until a Variable has no `graph_fn` (an input Variable set by the user) or a Variable with `require_grad` set to `False`, which helps save computations.


**Exercise**: compute the derivative with Keras, Tensorflow, CNTK, MXNet  

# Training loop

Let's take back our [Course 0](http://christopher5106.github.io/deep/learning/2018/10/20/course-zero-deep-learning.html)'s perceptron and implement its training directly with Pytorch tensors and operators, without other packages.
Let's consider the input is 20 dimensional, and the number of outputs for each dense layer is 32.

<img src="{{ site.url }}/img/deeplearningcourse/DL15.png">


Pytorch only requires to implement the forward pass of our perceptron. Each Dense layer is composed of two learnable parameters or weights:

```python
theta1 =  torch.autograd.Variable(torch.randn(32,20) *0.1,requires_grad = True)
bias1 = torch.autograd.Variable(torch.randn(32)*0.1,requires_grad = True)
theta2 = torch.autograd.Variable(torch.randn(32,32)*0.1,requires_grad = True)
bias2 = torch.autograd.Variable(torch.randn(32)*0.1,requires_grad = True)

def forward(x):
    # affine operation of the first Dense layer
    y = theta1.mv(x) + bias1
    # ReLu activation
    y = torch.max(y, torch.autograd.Variable(torch.Tensor([0])))
    # affine operation of the second Dense layer
    return theta2.mv(y) + bias2
```

As first loss function, let's use the square of the sum of the outputs. We can take whatever we want, as soon as it returns a scalar value to minimize:

```python
def cost(z):
    return (torch.sum(z)) ** 2
```

A training loop iterates over a dataset of training examples and each iteration consists in

- a forward pass : propagate the input values through layers from bottom to top, until the cost/loss

- a backward pass : propagate the gradients from top to bottom and into each of the parameters

- apply the parameter update rule $$ \theta \leftarrow \theta - \lambda \nabla_{\theta_L} \text{cost} $$ for each layer L

Let's train this network on random inputs, one sample at a time:

```python
for i in range(1000):
    lr = 0.001 * (.1 ** ( max(i - 500 , 0) // 100))

    x = torch.autograd.Variable(torch.randn(20), requires_grad=False)
    z = forward(x)
    c = cost(z)
    print("cost {} - learning rate {}".format(c.data.item(), lr))

    # compute the gradients
    c.backward()

    # apply the gradients
    theta1.data = theta1.data - lr * theta1.grad.data
    bias1.data = bias1.data - lr * bias1.grad.data
    theta2.data = theta2.data - lr * theta2.grad.data
    bias2.data = bias2.data - lr * bias2.grad.data

    # clear the grad
    theta1.grad.zero_()
    bias1.grad.zero_()
    theta2.grad.zero_()
    bias2.grad.zero_()
```

**Exercise**: check that the norm of the parameters converge to zero.

In a classification task,

- training is usually performed on batch of samples, instead of 1 sample, at each iteration, to get a faster training, but also to reduce the cost of the transfer of data when the data is moved to GPU

- the model is required to output a number of values equal to the number of classes, normalized by softmax activation

- the loss is the cross-entropy

as we have seen in [Course 0](http://christopher5106.github.io/deep/learning/2018/10/20/course-zero-deep-learning.html).

Let us consider a toy data, in which the label of a sample depends on its position in 2D, with 3 labels corresponding to 3 zones:


<img src="{{ site.url }}/img/deeplearningcourse/DL18.png">

The dataset creation or preprocessing is usually performed with Numpy:

```python
import matplotlib.pyplot as plt
dataset_size = 200000
x = np.random.rand(dataset_size, 2)
labels = np.zeros(dataset_size)
labels[x[:, 0] > x[:,1]] = 2
labels[x[:,1] + x[:, 0] > 1] = 1
plt.scatter(x[:,0], x[:,1], c=labels )
plt.show()
```

<img src="{{ site.url }}/img/deeplearningcourse/DL42.png">

Let's convert the Numpy arrays to Torch Tensors:

```python
X = torch.from_numpy(x).type(torch.FloatTensor)
Y = torch.from_numpy(labels).type(torch.LongTensor)
```

The input is defined by a position, ie a vector of dimension 2 for each sample, leading to a Tensor of size `(B, 2)`, where B is the batch size.

Let's choose as hidden dimension (number of outputs of first layer/ inputs of second layers) 12:

```python
theta1 =  torch.autograd.Variable(torch.randn(2, 12) *0.01,requires_grad = True)
bias1 = torch.autograd.Variable(torch.randn(12)*0.01,requires_grad = True)
theta2 = torch.autograd.Variable(torch.randn(12, 3)*0.01,requires_grad = True)
bias2 = torch.autograd.Variable(torch.randn(3)*0.01,requires_grad = True)

def forward(x):
    y = x.mm(theta1) + bias1 # (B, 2) x (2, 12) + (B, 12) => (B, 12)
    y = torch.max(y, torch.autograd.Variable(torch.Tensor([0])))
    y = y.mm(theta2) + bias2 # (B, 12) x (12, 3) + (B, 3) => (B, 3)
    return y

def softmax(z):
    e = torch.exp(z)
    s = torch.sum(e, 1, keepdim = True)
    return e / s

def crossentropy(s, l):
    v = torch.gather(s, 1, torch.unsqueeze(l,-1))
    v = torch.log(v)
    return -torch.mean(v)
```

For more efficiency, let's train 20 samples at each step, hence a batch size of 20:

```python
batch_size = 20
for i in range(min(dataset_size, 100000) // batch_size ):
    lr = 0.5 * (.1 ** ( max(i - 100 , 0) // 1000))

    batch = X[batch_size*i:batch_size*(i+1)] # size (batchsize, 2)
    z = forward(torch.autograd.Variable(batch, requires_grad=False))
    z = softmax(z)
    loss = crossentropy(z, torch.autograd.Variable(Y[batch_size*i:batch_size*(i+1)]))
    print("iter {} - cost {} - learning rate {}".format(i, c.data.item(), lr))

    # compute the gradients
    loss.backward()

    # apply the gradients
    theta1.data.sub_( lr * theta1.grad.data )
    bias1.data.sub_( lr * bias1.grad.data)
    theta2.data.sub_( lr * theta2.grad.data )
    bias2.data.sub_( lr * bias2.grad.data )

    # clear the grad
    theta1.grad.zero_()
    bias1.grad.zero_()
    theta2.grad.zero_()
    bias2.grad.zero_()
# iter 4999 - cost 0.07717917114496231 - learning rate 5.000000000000001e-05
```

The network converges.

When `loss.backward()` is called, the whole graph is differentiated w.r.t. the loss, and all Variables in the graph will have their .grad Variable accumulated with the gradient.

```python
print(loss.grad_fn)  # NegBackward
print(loss.grad_fn.next_functions[0][0])  # MeanBackward1
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # LogBackward
print(loss.grad_fn.next_functions[0][0].next_functions[0][0].next_functions[0][0])  # GatherBackward
```

To check everything is fine, one might compute the accuracy, a classical metric for classification problems:

```python
accuracy = 0
nb = 10000
for i in range(min(dataset_size, nb)):
    z = forward(torch.autograd.Variable(X[i:i+1], requires_grad=False))
    p = softmax(z)
    l = torch.max(p, -1)[1]
    if l.data.numpy()[0] == labels[i]:
        accuracy += 1

print("accuracy {}%".format(round(accuracy / min(dataset_size, nb) * 100,2)))
# accuracy 99.46%
```

**Exercise**: compute precision, recall, and AUC.

Note that convergence is strongly influenced

- by the art of choosing the learning rate
<img src="{{ site.url }}/img/deeplearningcourse/DL20.png">


- by the art of choosing the right layer initialization: a small variance, with positive and negative values to dissociate the neural outputs (neurons that fire together wire together) helps. In fact, we'll see in the next section Pytorch packages that provide a correct implementation of the variance choice given the number of input and output connections:
<img src="{{ site.url }}/img/deeplearningcourse/DL32.png">

To improve the results, it is possible to train multiple times the network from scratch, and average the ensemble of parameters from each training.

Note also that, if there is a backward function for every operation, there is no forward function: evaluation is performed when the operator is applied to the variable as in a classical program. You can still create your own function as in a classical program. The backward function works as a kind of history of the operations in order to retropropagate the gradient. So, it is very different from the concept of "graph of operators".

**Exercise**: program a training loop with Keras, Tensorflow, CNTK, MXNet


# Modules

A module is an object that encapsulates learnable parameters and is specifically suited to design deep learning neural networks.

A layer is a module:

<img src="{{ site.url }}/img/deeplearningcourse/DL31.png">

because it has weights and a forward function.

The composition of modules makes a module:

<img src="{{ site.url }}/img/deeplearningcourse/DL30.png">

The modules help organize layers and reuse their definitions.

move to GPU
exporting, loading

1- rewrite the model as module using nn modules

```python
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class SimpleNetTest(nn.Module):

    def __init__(self):
        super(SimpleNetTest, self).__init__()
        self.lin1 = nn.Linear(2, 12)
        self.lin2 = nn.Linear(12, 3)

    def forward(self, input):
        x = F.relu(self.lin1(input))
        x = self.lin2(x)
        return x       

net = SimpleNetTest()

# to move network to GPU
net.cuda()

print(net)
```

The learnable parameters are returned by net.parameters()

```python
params = list(net.parameters())
print(params)

SimpleNetTest(
  (lin1): Linear(in_features=2, out_features=12, bias=True)
  (lin2): Linear(in_features=12, out_features=3, bias=True)
)

```

```python
import torch.optim as optim

criterion = nn.CrossEntropyLoss()

# try various optimizer

# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# optimizer = torch.optim.Adam(net.parameters())
optimizer = torch.optim.Adadelta(net.parameters())

loss_curve = list()

batch_size = 500
for i in range(min(dataset_size, 1000000) // batch_size ):

    batch = x[batch_size*i:batch_size*(i+1)] # size (batchsize, 2)

    batch = torch.autograd.Variable(batch.cuda(), requires_grad=False)
    batchLabel = torch.autograd.Variable(labels[batch_size*i:batch_size*(i+1)].cuda())

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward
    outputs = net(batch)

    # backward
    loss = criterion(outputs, batchLabel)
    loss.backward()

    # update network parameters
    optimizer.step()

    # print("iter {} - cost {}".format(i, loss.data[0]))

    loss_curve.append(loss.data[0])

print("final cost {}".format(round(loss.data[0], 2)))
# final cost 0.16
```


```python
accuracy = 0
nb = 1000
for i in range(min(dataset_size, nb)):
    z = net(torch.autograd.Variable(x[i:i+1].cuda(), requires_grad=False))
    l = torch.max(z, -1)[1]
    if l.data.cpu().numpy()[0] == labels[i]:
        accuracy += 1

print("accuracy {}%".format(round(accuracy / min(dataset_size, nb) * 100, 2)))
# accuracy 95.40
```

```python
accuracy = 0
z = net(torch.autograd.Variable(x.cuda(), requires_grad=False))

l = torch.max(z, 1)[1]
ll = torch.autograd.Variable(labels.cuda())

accuracy = int(torch.sum(torch.eq(l, ll).type(torch.cuda.LongTensor)))
print("accuracy {}%".format(accuracy / dataset_size * 100))
```



```python
plt.plot(range(1, len(nplc)+1), nplc, 'ro')
```


<img src="{{ site.url }}/img/deeplearningcourse/DL43.png">

**Exercise**: program with packages in Keras, Tensorflow, CNTK, MXNet

# Packages

Packages help you reuse common functions for deep learning.


2- rewrite training loop using the optim package (zeroing gradients + applying the gradients with an update rule)
3- look at different update rules
4- plot the training curves (loss,...)
5- gpu


view reshape

**Exercise**: replace your functions with package functions in Keras, Tensorflow, CNTK  

**Well done!**

Now let's go to next course: [Course 2: building deep learning networks!](http://christopher5106.github.io/deep/learning/2018/10/20/course-two-build-deep-learning-networks.html)
