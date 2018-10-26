from keras import backend as K
import numpy as np

dataset_size = 200000
X = np.random.rand(dataset_size, 2)
labels = np.zeros((dataset_size, 3))
labels[X[:, 0] > X[:,1]] = [0,0,1]
labels[X[:, 0] <= X[:,1]] = [1,0,0]
labels[X[:,1] + X[:, 0] > 1] = [0, 1, 0]


x = K.placeholder(shape=(None, 2))
t = K.placeholder(shape=(None, 3))

theta1 = K.random_normal_variable(shape=(2, 12), mean=0, scale=0.01)
bias1 = K.random_normal_variable(shape=(1, 12), mean=0, scale=0.01)

theta2 = K.random_normal_variable(shape=(12, 3), mean=0, scale=0.01)
bias2 = K.random_normal_variable(shape=(1, 3), mean=0, scale=0.01)

#theta1 = K.variable(value=np.random.normal(scale = 0.01, size=(2,12)))
#bias1 = K.variable(value=np.random.normal(scale = 0.01, size=(1,12)))

#theta2 = K.variable(value=np.random.normal(scale = 0.01, size=(12,3)))
#bias2 = K.variable(value=np.random.normal(scale = 0.01, size=(1,3)))


def forward(x):
    y = K.dot(x, theta1) + bias1
    y = K.maximum(y, 0.)
    return K.dot(y, theta2) + bias2

def softmax(x):
    e = K.exp(x)
    s = K.sum(e, axis=1, keepdims=True)
    return e/s

def crossentropy(y, t):
    prob = K.sum(y*t, axis=1)
    return - K.mean(K.log(prob))

loss = crossentropy(softmax(forward(x)),t)
params= [theta1, bias1, theta2, bias2]
grad = K.gradients(loss, params)
f = K.function([x,t], [loss]+grad)

batch_size = 20
for i in range(min(dataset_size, 100000) // batch_size ):
    lr = 0.5 * (.1 ** ( max(i - 100 , 0) // 1000))
    sample = X[batch_size*i:batch_size*(i+1)]
    target = labels[batch_size*i:batch_size*(i+1)]
    outputs = f([sample, target])
    for param,grad in zip(params, outputs[1:]):
        K.set_value(param, K.eval(param) - grad * lr)
    print("cost {} - learning rate {}".format(outputs[0], lr))


f = K.function([x], [K.argmax(forward(x),axis=1)])
accuracy = 0
for i in range(1000):
    sample = X[batch_size*i:batch_size*(i+1)]
    target = labels[batch_size*i:batch_size*(i+1)]
    tt = f([sample])[0]
    accuracy += np.sum(tt == np.argmax(target, axis=1))

print("Accuracy", accuracy / 1000. /batch_size)
# accuracy 99.44
