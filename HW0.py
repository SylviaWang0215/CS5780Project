import time

import numpy as np

def l2distance(X, Z = None):
    if Z is None:
        Z = X
    size_x = len(X)
    size_z = len(Z)
    temp1 = np.dot(X, X.T).diagonal()
    #print "diagonal shape"
    #print temp1.shape
    temp1 = np.asmatrix(temp1)
    #temp3 = np.dot(X, X.T).diagonal().reshape(size_x, 1)
    #print temp1.shape
    #S = temp3*np.ones((1, size_z))
    S = (np.ones((size_z, 1)) * temp1).T
    #print S2
    #print S
    #temp2 = np.dot(Z, Z.T).diagonal().reshape(size_z, 1)
    temp2 = np.asmatrix(np.dot(Z, Z.T).diagonal())
    #print temp2.shape
    R = np.ones((size_x, 1))*temp2
    G = np.dot(X, Z.T)
    D = np.sqrt(R + S - 2*G)

    r1, r2 = R.shape
    #print r1, r2
    print D
    return D


M = np.array([[1,2,3],[4,5,6],[7,8,9]])
Q = np.array([[11,12,13],[14,15,16]])

def innerproduct(X, Z = None):
    if Z is None:
        Z = X
    G = np.dot(X, Z.T)
    return G


def l2distanceSlow(X, Z=None):
    if Z is None:
        Z = X

    n, d = X.shape  # dimension of X
    m = Z.shape[0]  # dimension of Z
    D = np.zeros((n, m))  # allocate memory for the output matrix
    for i in range(n):  # loop over vectors in X
        for j in range(m):  # loop over vectors in Z
            D[i, j] = 0.0;
            for k in range(d):  # loop over dimensions
                D[i, j] = D[i, j] + (X[i, k] - Z[j, k]) ** 2;  # compute l2-distance between the ith and jth vector
            D[i, j] = np.sqrt(D[i, j]);  # take square root
    return D
''''
current_time = lambda: int(round(time.time() * 1000))

X=np.random.rand(700,100)
Z=np.random.rand(300,100)

print("Running the naive version ...")
before = current_time()
D=l2distanceSlow(X)
after = current_time()
t_slow = after - before

print("Running the vectorized version ...")
before = current_time()
D=l2distance(X)
after = current_time()
t_fast = after - before

speedup = t_slow / t_fast
print("The numpy code was {} times faster!".format(speedup))
'''

x = np.matrix('-4; -3; -2; 2; 3; 4')
y = np.matrix('0;1')
D = l2distance(x, y)
#print D[0, 0] == 0