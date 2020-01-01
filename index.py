from __future__ import division

import numpy, activation
from data import X_train, X_test, y_train, y_test

# We use this function to get a tensor of initial
# random weights, given the size of the input
# layer, the number and size of the hidden layer(s)
# and the size of the output layer:
def init_weights(ninput, nhidden, noutput):
    # ninput: scalar representing the input layer size.
    # nhidden: scalar or list representing the number
    #          and size of the hidden layer(s).
    # noutput: scalar representing the output layer size
    if type(nhidden) is int:
        nhidden=[nhidden]

    res=[]

    for l in range(len(nhidden)):
        res.append([])
        for nodes in range(nhidden[l]):
            res[l].append([])
            for i in range(ninput):
                res[l][nodes].append(numpy.random.ranf())
        ninput=len(res[l])

    res.append([])

    res[len(res)-1]=[[] for i in range(noutput)]

    for i in range(len(res[len(res)-2])):
        for j in range(noutput):
            res[len(res)-1][j].append(numpy.random.ranf())

    return [numpy.array(i, numpy.dtype('Float64')) for i in res]

# A feed-forward neural network, given the tensor
# of weights "W" and the input information "x":
def feedforward(W, x, transfer='softmax'): 
    # W: tensor of weights.
    # x: input information
    # transfer: activation function to use.

    nodes=[x]

    for w in W:
        net=list()
        for node in w:
            net.append(numpy.sum(w*x))

        x=getattr(activation, transfer)(net)

        nodes.append(x)

    return {'last_layer': nodes[-1], 'lower_layers': nodes[:-1]}
    # the last element in list "nodes"
    # is the output result, that is
    # the output layer. "lower layers"
    # are the input layer along with
    # the hidden layer(s).

# We use this function to get the gradient of the error with
# respect to every single weight in the network using the
# backward propagation of error algorithm. Notice how the error 
# is obtained with the difference between the output vector and 
# the target vector. Then this error is "propagated" being part
# of the operation to get every gradient:
def backpropagation(output_vector, target_vector, lower_layers, W):
    # discrepacy between the output vector
    # and the target vector:
    delta_err=output_vector - target_vector

    gradients=[]

    for j in reversed([i for i in range(len(W[1:]))]):
        gradients.append( numpy.array( [ sum( [ delta_err[i] * W[j+1][i] for i in range( len( delta_err ) ) ] ) * lower_layers[j+1]*( 1 - lower_layers[j+1] ) * h for h in lower_layers[j] ], dtype=numpy.dtype('Float64')).transpose() )

    gradients.insert( 0, numpy.array( [ delta_err * i for i in lower_layers[-1] ], dtype=numpy.dtype('Float64') ).transpose() )

    return gradients

# This is the training function, which consists of adjusting
# or updating the weights, using stochastic gradient descent
# given the training dataset "X" and the target vector dataset
# "y". Also is given the number of epochs and a learning rate:
def fit(X, y, nhidden, noutput, epochs=10, learning_rate=1):
    ninput=len(X[0])
    # epochs: number of times the entire dataset X
    #         is used in this training phase.
    # learning rate: step size at each iteration.
    # nhidden: scalar or list representing the number
    #          and size of the hidden layer(s).
    # noutput: scalar representing the output layer size.
    W=init_weights(ninput, nhidden, noutput)

    for epoch in range(epochs):
        for i, x in enumerate(X):
            nodes=feedforward(W, x)

            output_vector=nodes['last_layer']
            target_vector=y[i]
            lower_layers=nodes['lower_layers']

            weight_gradients=backpropagation(output_vector, target_vector, lower_layers, W)

            # weights update:
            for j,g in enumerate(reversed(range(len(W)))):
                W[j]=W[j] - learning_rate * weight_gradients[g]

    return W

# Given the testing dataset "X", its
# corresponding targetvector dataset
# and the and the trained/adjusted weights,
# this function tests the algorithm accuracy: 
def score(X, y, W):
    accuracy=0

    for i, x in enumerate(X):
        # Simply the feedforfard function is
        # used to get the output layer
        output=feedforward(W, x)['last_layer']
        y_hat=list(output).index(max(output))
        y_true=list(y[i]).index(max(y[i]))

        if y_hat!=y_true: accuracy+=1

    return accuracy/len(y)

trained_weights=fit(X_train, y_train, 52, 10)

print(score(X_test, y_test, trained_weights))
