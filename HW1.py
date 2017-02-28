import numpy as np
import sys
import matplotlib
matplotlib.use("PDF")
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.stats import mode
import scipy
import time
import operator


def loaddata(filename):
    """
    Returns xTr,yTr,xTe,yTe
    xTr, xTe are in the form nxd
    yTr, yTe are in the form nx1
    """
    data = loadmat(filename)
    xTr = data["xTr"] # load in Training data
    yTr = np.round(data["yTr"]) # load in Training labels
    xTe = data["xTe"] # load in Testing data
    yTe = np.round(data["yTe"]) # load in Testing labels
    return xTr.T, yTr.T, xTe.T, yTe.T


def innerproduct(X, Z = None):
    if Z is None:
        Z = X
    G = np.dot(X, Z.T)
    return G


def l2distance(X, Z = None):
    if Z is None:
        Z = X
    size_x = len(X)
    size_z = len(Z)
    temp1 = np.dot(X, X.T).diagonal().reshape(size_x, 1)
    S = temp1*np.ones((1, size_z))
    temp2 = np.dot(Z, Z.T).diagonal().reshape(size_z, 1)
    R = np.ones((size_x, 1))*temp2.T
    G = np.dot(X, Z.T)
    D = np.sqrt(R + S - 2*G)

    r1, r2 = R.shape
    return D


def findknn(xTr, xTe, k):
    """
    function [indices,dists]=findknn(xTr,xTe,k);

    Finds the k nearest neighbors of xTe in xTr.

    Input:
    xTr = nxd input matrix with n row-vectors of dimensionality d
    xTe = mxd input matrix with m row-vectors of dimensionality d
    k = number of nearest neighbors to be found

    Output:
    indices = kxm matrix, where indices(i,j) is the i^th nearest neighbor of xTe(j,:)
    dists = Euclidean distances to the respective nearest neighbors
    """
    temp_dis = l2distance(xTr, xTe)
    indices = np.argsort(temp_dis, axis=0)[0:k, :]
    dists = np.sort(temp_dis, axis=0)[0:k, :]

    return indices, dists



def analyze(kind, truth, preds):
    """
    function output=analyze(kind,truth,preds)
    Analyses the accuracy of a prediction
    Input:
    kind='acc' classification error
    kind='abs' absolute loss
    (other values of 'kind' will follow later)
    """

    truth = truth.flatten()
    preds = preds.flatten()
    c = truth.size
    print c

    if kind == 'abs':
        return (abs(truth - preds).sum() + 0.0) / c
    elif kind == 'acc':
        return (np.sum(truth == preds) + 0.0) / c


def knnclassifier(xTr, yTr, xTe, k):
    """
    function preds=knnclassifier(xTr,yTr,xTe,k);

    k-nn classifier

    Input:
    xTr = nxd input matrix with n row-vectors of dimensionality d
    xTe = mxd input matrix with m row-vectors of dimensionality d
    k = number of nearest neighbors to be found

    Output:

    preds = predicted labels, ie preds(i) is the predicted label of xTe(i,:)
    """

    if k == 1 and np.array_equal(xTe, xTr):
        preds = yTr
        return preds
    print("xTr")
    print(xTr)
    print("yTr")
    print(yTr)
    print("xTe")
    print(xTe)
    r = yTr.shape[0]
    lenTe = len(xTe)
    lenTr = len(xTr)
    yTr = yTr.reshape((1, lenTr))
    indices, dist = findknn(xTr, xTe, k)
    preds = np.zeros((lenTe, 1))
    lisyTr = yTr.tolist()
    lisindices = (indices.T).tolist()
    for i in range(lenTe):
        temp_lis = [lisyTr[0][item] for item in lisindices[i]]
        new_list = sorted(temp_lis, key=temp_lis.count, reverse=True)
        preds[i, 0] = new_list[0]

    if r == 1:
        preds = preds.reshape((1, lenTe))
    return preds

def competition(xTr, yTr, xTe):
    """
    function preds=competition(xTr,yTr,xTe);

    A classifier that outputs predictions for the data set xTe based on
    what it has learned from xTr,yTr

    Input:
    xTr = nxd input matrix with n column-vectors of dimensionality d
    xTe = mxd input matrix with n column-vectors of dimensionality d

    Output:

    preds = predicted labels, ie preds(i) is the predicted label of xTe(i,:)
    """
    lenTr = len(xTr)
    print("length of training dataset")
    print(lenTr)
    lenTe = len(xTe)
    # use group of data to find the optimal k
    sub_xTe = xTr[0: lenTr / 10, :]
    sub_yTe = yTr[0: lenTr / 10, :]
    sub_xTr = xTr[lenTr / 10:, :]
    sub_yTr = yTr[lenTr / 10:, :]
    sub_lenTr = len(sub_xTr)
    acc = 0
    optimal_k = 1
    for i in range(1, sub_lenTr):
        temp_preds = knnclassifier(sub_xTr, sub_yTr, sub_xTe, i)
        temp_acc = analyze("acc", sub_yTe, temp_preds)
        if temp_acc > acc:
            optimal_k = i
            acc = temp_acc
    preds = knnclassifier(xTr, yTr, xTe, optimal_k)
    print "the optiaml k"
    print optimal_k
    return preds

a = np.matrix('0 0; 1 0; 2 0; 3 0; 0 1; 2 1; 3 1; 0 2; 1 2; 0 3; 3 3')
a_label = np.matrix('0 0 1 1 0 1 1 1 1 1 1').T
b = np.matrix('-1 0; 2 3')
x, y = findknn(a, b, 4)

#mat1 = np.matrix('1 2 1 1')
#mat2 = np.matrix('1 2 1 2')
#print analyze('abs', mat1, mat2)

linear_x = np.matrix('-4 -3 -2 2 3 4')
linear_y = np.matrix('1 1 1 2 2 2')

#x = knnclassifier(a, a_label, a, 1)
#print "x"
#print x
#print analyze('acc', a_label, x)


x_tr = np.matrix('0 0; 0 1; 1 0; 1 1')
y_tr = np.matrix('1; 2; 2; 1')
x_te = np.matrix('0 0; 0 1; 1 0; 1 1')
test =  y_tr.tolist()
print test

print len(x_te)
#preds=knnclassifier(x_tr, y_tr, x_te, 2)
#print(preds)

competition(x_tr, y_tr, x_te)