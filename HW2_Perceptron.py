
import numpy as np
from numpy.matlib import repmat

import sys
import matplotlib.pyplot as plt
from scipy.io import loadmat
import time

def loaddata(filename):
    """
    Returns xTr,yTr,xTe,yTe
    xTr, xTe are in the form nxd
    yTr, yTe are in the form nx1
    """
    data = loadmat(filename)
    xTr = data["xTr"]; # load in Training data
    yTr = np.round(data["yTr"]); # load in Training labels
    xTe = data["xTe"]; # load in Testing data
    yTe = np.round(data["yTe"]); # load in Testing labels
    return xTr.T,yTr.T,xTe.T,yTe.T


def row_vectorize(x):
    return x.reshape(1,-1)


def perceptronUpdate(x, y, w):
    """
    function w=perceptronUpdate(x,y,w);

    Implementation of Perceptron weights updating
    Input:
    x : input vector of d dimensions (1xd)
    y : corresponding label (-1 or +1)
    w : weight vector before updating

    Output:
    w : weight vector after updating
    """
    # just in case x, w are accidentally transposed (prevents future bugs)
    x, w = map(row_vectorize, [x, w])  # map is for return a list
    assert(y in {-1,1})
    print("w.shape")
    print(w.shape)
    print("x.shape")
    print(x.shape)
    w = w + y * x
    ## fill in code here
    return w



#xTr,yTr,xTe,yTe=loaddata("../resource/lib/digits.mat")

def perceptron(x, y):
    """
    function w=perceptron(x,y);

    Implementation of a Perceptron classifier
    Input:
    x : n input vectors of d dimensions (nxd)
    y : n labels (-1 or +1)

    Output:
    w : weight vector (1xd)
    """

    n, d = x.shape
    w = np.zeros(d)
    count = 0
    while count < 100:
        m = 0
        for i in range(n):
            x_temp = x[i]
            y_temp = y[i]
            if (y_temp * np.dot(w, x_temp.T) <= 0).all():  # dot(w, x.T) return a matrix type
                w = perceptronUpdate(x_temp, y_temp, w)
                m += 1
        count += 1
        if m == 0:
            break
    w = np.asarray(w).reshape(-1)
    return w


def classifyLinear(x, w, b=0):
    """
    function preds=classifyLinear(x,w,b)

    Make predictions with a linear classifier
    Input:
    x : n input vectors of d dimensions (dxn)
    w : weight vector (dx1)
    b : bias (scalar)

    Output:
    preds: predictions (1xn)
    """
    w = w.reshape(-1)
    n, d = x.shape
    preds = np.zeros(n)
    for i in range(n):
        x_temp = x[i]
        if (np.dot(w, x_temp.T) + b < 0).all():  # if wx + b < 0 is true, then label should be -1
            preds[i] = -1
        else:
            preds[i] = 1
    return preds


N = 100
# Define the symbols and colors we'll use in the plots later
symbols = ['ko', 'kx']
mycolors = [[0.5, 0.5, 1], [1, 0.5, 0.5]]
classvals = [-1, 1]

# generate random (linarly separable) data
trainPoints = np.random.randn(N, 2) * 1.5

# defining random hyperplane
w = np.random.rand(2)

# assigning labels +1, -1 labels depending on what side of the plane they lie on
trainLabels = np.sign(np.dot(trainPoints, w))
i = np.random.permutation([i for i in range(N)])

# shuffling training points in random order
trainPoints = trainPoints[i, :]
trainLabels = trainLabels[i]

# call perceptron to find w from data
w = perceptron(trainPoints,trainLabels)
b = 0

res=300
xrange = np.linspace(-5, 5,res)
yrange = np.linspace(-5, 5,res)
pixelX = repmat(xrange, res, 1)
pixelY = repmat(yrange, res, 1).T

testPoints = np.array([pixelX.flatten(), pixelY.flatten(), np.ones(pixelX.flatten().shape)]).T
testLabels = np.dot(testPoints, np.concatenate([w.flatten(), [b]]))

Z = testLabels.reshape(res,res)
plt.contourf(pixelX, pixelY, np.sign(Z), colors=mycolors)
plt.scatter(trainPoints[trainLabels == classvals[0],0],
            trainPoints[trainLabels == classvals[0],1],
            marker='o',
            color='k'
           )
plt.scatter(trainPoints[trainLabels == classvals[1],0],
            trainPoints[trainLabels == classvals[1],1],
            marker='x',
            color='k'
           )
#plt.quiver(0, 0, w[0, 0], w[0, 1], linewidth=0.5, color=[0, 1, 0])
plt.axis('tight')
plt.show()