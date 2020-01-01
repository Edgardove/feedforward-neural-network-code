import numpy

def softmax(x):
    e_x = numpy.exp(x - numpy.max(x))
    return e_x / e_x.sum()

def sigmoid(x):
    x=numpy.array(x)
    return numpy.where(x >= 0, 1 / (1 + numpy.exp(-x)), numpy.exp(x) / (1 + numpy.exp(x)))

def relu(x):
    x=x/numpy.max(x)
    return numpy.maximum(x,0)