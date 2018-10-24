import cntk as C
import numpy as np

dataset_size = 200000
X = np.random.rand(dataset_size, 2)
labels = np.zeros((dataset_size, 3))
labels[X[:, 0] > X[:,1]][2] = 1
labels[X[:,1] + X[:, 0] > 1][1] = 1
labels[:, 0] = 1- labels[:,1] - labels[:,2]

x = C.input_variable(shape=(2,), needs_gradient=False)
t = C.input_variable(shape=(3), needs_gradient=False)

init = C.initializer.normal(0.01)

theta1 = C.Parameter(shape=(2, 12), init=init )
bias1 = C.Parameter(shape=(12,), init=init )

theta2 = C.Parameter(shape=(12,3), init=init )
bias2 = C.Parameter(shape=(3,), init=init )

def forward(x):
    y = C.times(x, theta1) + bias1
    y = C.element_max(y, 0.)
    return C.times(y, theta2) + bias2

y = C.cross_entropy_with_softmax(forward(x), t)

for i in range(1000):
    lr = 0.01 * (.1 ** ( max(i - 500 , 0) // 100))
    sample = X[i]
    target = labels[i]
    g = y.grad({x:sample, t:target}, wrt=[theta1, bias1, theta2, bias2])
    for param,grad in g.items():
        param.value = param.value - grad * lr
    loss = y.eval({x:sample, t:target})
    print("cost {} - learning rate {}".format(loss[0][0], lr))
