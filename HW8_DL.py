
# coding: utf-8

# <h2>Project 8: Deep Learning</h2>
# 
# 
# <h3>Introduction</h3>
# In this project, you will implement a neural network.
# We broke it apart into several functions and a pre-processing step.</p>
# 

# In[5]:

#<GRADED>
import numpy as np
from numpy.matlib import repmat
#</GRADED>
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
import sys
from scipy.io import loadmat
import time

# add p08 folder
sys.path.insert(0, './p08/')

get_ipython().magic('matplotlib inline')


#  <p> 
#   An important step to make neural networks function properly is to scale the data properly. Therefore, first implement the preprocess function
#   It takes as input the training and the test data and transforms them both with the same mapping: $\vec x_i\rightarrow U\vec x_i-\vec{m}$. 
#   After this transformation the training data set should have zero-mean and each feature should have standard deviation 1 . The same transformation is then also applied to the test data (Hint: The matrix U is typically diagonal.) 
#   <!--	 <li>HINT 2: Ideally you would like the input features to be de-correlated. The correlation matrix should be diagonal (in this case even the identity matrix). One way to do this is to project the data onto the PCA principal components (which we will still cover later in the course). You can get the transposed projection matrix by calling $pcacov(xTr')$. Make sure to apply PCA <i>after</i> you subtracted off the mean. </li>-->
#   </p>

# In[6]:

#<GRADED>
def preprocess(xTr, xTe):
    """
    Preproces the data to make the training features have zero-mean and
    standard-deviation 1
    INPUT:
        xTr - nxd training data
        xTe - mxd testing data
    OUPUT:
        xTr - pre-processed training data
        xTe - pre-processed testing data
        s,m - standard deviation and mean of xTr
            - any other data should be pre-processed by x-> (x-m)/s
    (The size of xTr and xTe should remain unchanged)
    """
    
    ntr, d = xTr.shape
    nte, _ = xTe.shape
  
    ## TODO 1  
    for i in range(d):
        xi = xTr[:,i]
        x_m = xi.mean()
        x_s = xi.std()
        xTr[:,i] = (xi-x_m)/x_s
    
    for i in range(d):
        xi = xTe[:,i]
        x_m = xi.mean()
        x_s = xi.std()
        xTe[:,i] = (xi-x_m)/x_s

    ## TODO 1
    m = xTr.mean()
    s = xTr.std()
    
    
    return xTr, xTe, s, m
#</GRADED>


# <p> 
#   Now you need to implement three transition functions: <code>ReLU, sigmoid</code> and <code>tanh</code>.
#  </p>
#  <ol>
#      <li>ReLU: $\sigma(z) = \max(0, z)$</li>
#      <li>Sigmoid: $\sigma(z) = \frac{1}{1 + e^{-z}}$</li>
#      <li>tanh: $\sigma(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$</li>
#  </ol>

# In[7]:

#<GRADED>
def get_transition_func(transtype):
    """
    Given the type, gets a specific transition function
    INPUT:
        transtype - "sigmoid", "tanh", "ReLU", "sin"
    OUTPUT:
        trans_func - transition function (function)
        trans_func_der - derivative of the transition function (function)

    (type must be one of the defined transition functions)
    """
    
    assert transtype in ["sigmoid", "tanh", "ReLU","sin"]
    if transtype == "sin":   
        trans_func = lambda z: np.sin(z)
        trans_func_der = lambda z: np.cos(z)
    ## TODO 2
    
    elif transtype == "ReLU":
        trans_func = lambda z: np.maximum(z, 0)
        trans_func_der = lambda z : np.divide(np.maximum(z, 0), z)
    elif transtype == "tanh":
        trans_func = lambda z: (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
        trans_func_der = lambda z: 1 - ((np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z)))**2
    else:
        trans_func = lambda z: 1/(1 + np.exp(-z))
        trans_func_der = lambda z: np.exp(-z) / (1 + np.exp(-z))**2
        
    ## TODO 2
    
    return trans_func, trans_func_der
#</GRADED>


# <p>If everything is correct we can now plot these functions and check their gradients. The gradient errors should all be very small (less than $10^{-10}$.) </p>

# In[8]:

def numericalgradient(fun,x,e):
    dh = 0
    nx = x    # copy the weight vector
    nx += e  # perturn dimension i
    l1 = fun(nx) # compute loss
    nx -= 2*e # perturm dimension i again
    l2 = fun(nx) # compute loss
    dh = (l1 - l2)/(2*e) # the gradient is the slope of the loss
    return dh


# In[9]:

fig, axarr = plt.subplots(1, 4, sharex=True, sharey=True)
fig.set_figwidth(15)
fig.set_figheight(3)
names = ["sigmoid","tanh","ReLU","sin"]

for idx, name in enumerate(names):
    # plot stuff
    f, delta_f = get_transition_func(name)
    x = np.arange(-5, 5, 0.1)
    axarr[idx].plot(x, f(x))
    
    
    axarr[idx].axis([-5,5,-1,1])
    axarr[idx].title.set_text(name)
    axarr[idx].grid(True)
    
    # check gradients
    print("%s gradient check at x=1: " % name, end='')
    dh = numericalgradient(f,1,1e-5)
    dy = delta_f(1)
    num = np.linalg.norm(dh-dy)
    denom = np.linalg.norm(dh+dy)
    graderror = num/denom if denom != 0 else 0
    if graderror < 1e-10:
        print("passed ", end='')
    else:
        print("FAILED ", end='')
    print('at x=-1: ', end='')
    dh2 = numericalgradient(f,-1,1e-5)
    dy2 = delta_f(-1)
    num = np.linalg.norm(dh2-dy2)
    denom = np.linalg.norm(dh2+dy2)
    graderror += num/denom if denom != 0 else 0
    if graderror < 1e-10:
        print("passed")
    else:
        print("FAILED")


# <p>
# The following function will randomly generate initial weights for the neural network.
# </p>

# In[10]:

def initweights(specs):
    """
    Given a specification of the neural network, output a random weight array
    INPUT:
        specs - array of length m+1
    
    OUTPUT:
        weights - array of length m, each element is a matrix
            where size(weights[i]) = (specs[i], specs[i+1])
    """
    weights = []
    for i in range(len(specs) - 1):
        weights.append(np.random.randn(specs[i], specs[i+1]))
    return weights


# 
#   <li><p> Now implement the forward pass function
#   <pre>forward_pass(W,xTr,trans_func)</pre>
#   It takes the weights for the network, the training data, and the transition function to be used between layers.  It should output the result at each node for the forward pass. Each layer has two outputs, $A[i]$ and $Z[i]$, where $A[i]=Z[i-1] * weights[i-1]$ and $Z[i]=trans\_func(A[i])$. $weights[0]$ stores the weights for the first layer of the network.
#   </p>
#   </li>
# 
# 

# In[11]:

#<GRADED>
def forward_pass(weights, xTr, trans_func):
    """
    INPUT:
        weights - weights (cell array of length m)
        xTr - nxd matrix (each row is an input vector)
        trans_func - transition function to apply for inner layers
    
    OUTPUTS:
        A, Z - result of forward pass (cell array of length m+1)
    
    Hint:
        Make sure A[0]=Z[0]=xTr and A[m] = Z[m] (Why?)
    """
    
    ## TODO 3   
    A, Z = [], []
    A.append(xTr)
    Z.append(xTr)
    m = len(weights)
    
    temp_z = Z[0]
    for i in range(m-1):
      
        temp_a = np.dot(temp_z, weights[i])
        A.append(temp_a)            
        temp_z = trans_func(temp_a)       
        Z.append(temp_z)
    temp_a = Z[-1].dot(weights[-1])
    A.append(temp_a)
    Z.append(temp_a)
    ## TODO 3
    return A, Z
#</GRADED>


# In[12]:

weights=initweights([2,20,20,1])
f,delta_f=get_transition_func("sigmoid")
x=np.random.rand(100,2);
A,Z=forward_pass(weights,x,f)


#   <li><p> Now compute the loss for the network
#   <pre>compute_loss(zs, yTr)</pre>
#   It takes the output of the forward pass and the training labels.  It should compute the loss for the entire training set averaging over all the points:
#   $$L(x, y) = \frac{1}{2n}(H(x) - y)^2$$
#   </p>
#   </li>

# In[13]:

#<GRADED>
def compute_loss(Z, yTr):
    """
    INPUT:
        Z - output of forward pass (cell array of length m+1)
        yTr - array of length n
    
    OUTPUTS:
        loss - the average squared loss obtained with Z and yTr (scalar)
    """
    
    delta = Z[-1].flatten() - yTr.flatten()
    
    n = len(yTr)
    loss = 0

    ## TODO 4
    loss = 0.5/n * sum(delta**2)
    ## TODO 4

    return loss
#</GRADED>


# In[14]:

yTr = np.random.rand(100)
compute_loss(Z,yTr)


#   <li><p> Now implement <i>back propagation</i> 
#   <pre>backprop(W, as, zs, yTr,  der_trans_func)</pre>
#   to compute the gradient for the weights and bias terms. 
#   It takes the weights for the network, the outputs of the forward pass, the training labels, and the derivative of the transition function.  Use the chain rule to calculate the gradient of the weights.
#   </p></li>
#   		
# 
# 
# 

# In[21]:

#<GRADED>
def backprop(weights, A, Z, yTr, delta_f):
    """
    INPUT:
        weights - weights (cell array of length m)
        A - output of forward pass (cell array of length m+1)
        Z - output of forward pass (cell array of length m+1)
        yTr - array of length n
        delta_f - derivative of transition function to apply for inner layers
    
    OUTPUTS:
        gradient - the gradient at w (cell array of length m)
    """
    
    yTr = yTr.reshape(-1,1)
    n,_ = yTr.shape
    delta = (Z[-1].flatten() - yTr.flatten()).reshape(-1, 1)
    # compute gradient with back-prop
    gradients = []
    m = len(weights)
    for i in range(m, 0, -1):
        #temp_w = np.dot(np.transpose(delta), Z[i-1])    
        temp_w = np.dot(np.transpose(Z[i-1]),delta)/n
        gradients.append(temp_w)    
        #delta = np.multiply(delta_f(A[i-1]), np.dot(np.transpose(delta), weights[i])
        
        w_delta = np.dot(delta,weights[i-1].T)
        
        
        delta = np.multiply(delta_f(A[i-1]), w_delta)
        
    gradients = gradients[::-1]  
    print("the length of gradients:" + str(len(gradients)))
    ## TODO 5
    
    ## TODO 5
    return gradients
#</GRADED>


# <p>The following function will plot the results of prediction and the loss.</p>

# In[22]:

def plot_results(x, y, Z, losses):
    fig, axarr = plt.subplots(1, 2)
    fig.set_figwidth(12)
    fig.set_figheight(4)

    axarr[0].plot(x, y)
    axarr[0].plot(x, Z[-1].flatten())
    axarr[0].set_ylabel('$f(x)$')
    axarr[0].set_xlabel('$x$')
    axarr[0].legend(['Actual', 'Predicted'])

    axarr[1].semilogy(losses)
    axarr[1].title.set_text('Loss')
    axarr[1].set_xlabel('Epoch')

    plt.show()


# <p>
#     We can use the gradient from back propagation to update the weights in each layer. There are quite a few ways to do the update. Here we demonstrate three variations of gradient descent.
# </p>

# ## Momentum

# In[23]:

x = np.arange(0, 5, 0.1)
y = (x ** 2 + 10*np.sin(x))

x2d = np.concatenate([x, np.ones(x.shape)]).reshape(2, -1).T

weights = initweights([2,200,1])

print("the shape of weights[0]" + str(weights[0].shape))
print("the shape of weights[0]" + str(weights[1].shape))
print("the length of weights" + str(len(weights)))

momentum = np.copy(weights)
for j in range(len(weights)):
    momentum[j] = momentum[j] * 0
    
alpha = 0.01
M = 10000
beta = 0.8

losses = np.zeros(M)
t0 = time.time()
for i in range(M):
    f, delta_f = get_transition_func("ReLU")
    A, Z = forward_pass(weights, x2d, f)
    
    print("the shape of A[0]" + str(A[0].shape))
    print("the shape of A[0]" + str(A[1].shape))
    print("the shape of A[0]" + str(A[2].shape))
    print("the length of weights" + str(len(A)))
    
    print("the shape of Z[0]" + str(Z[0].shape))
    print("the shape of Z[0]" + str(Z[1].shape))
    print("the length of weights" + str(len(Z)))
    
    losses[i] = compute_loss(Z,y)
    gradients = backprop(weights,A,Z,y,delta_f)
    for j in range(len(weights)):
        momentum[j] = beta * momentum[j] + (1 - beta) * gradients[j]
        weights[j] -= alpha * momentum[j]
t1 = time.time()
print('Elapsed time: %.2fs (probably faster than Julia)' % (t1-t0))



# In[72]:

plot_results(x, y, Z, losses)


# <p>
#   If you did everything correctly, the result should look similar to this image:
# </p>
# <center>
#    <img src="./p8/plot.png" width="800px" />
# </center>

# ## Custom step-size decrease schedule 

# In[25]:

# training data
x = np.arange(0, 5, 0.1)
y = (x ** 2 + 10*np.sin(x))
x2d = np.concatenate([x, np.ones(x.shape)]).reshape(2, -1).T

# transition function
f, delta_f = get_transition_func("ReLU")

# initialize weights, historical gradients, losses
weights = initweights([2,200,1])
losses = np.zeros(M)

hist_grad = np.copy(weights)
for j in range(len(weights)):
    hist_grad[j] = hist_grad[j] * 0

alpha = 0.02
M = 10000
beta = 0.8
eps = 1e-6

losses = np.zeros(M)
t0 = time.time()
for i in range(M):
    f, delta_f = get_transition_func("ReLU")
    A, Z = forward_pass(weights, x2d, f)
    losses[i] = compute_loss(Z,y)
    gradients = backprop(weights,A,Z,y,delta_f)
    for j in range(len(weights)):
        if i != 0:
            hist_grad[j] = beta * hist_grad[j] + (1 - beta) * gradients[j] ** 2
        else:
            hist_grad[j] = gradients[j] ** 2
        adj_grad = gradients[j] / (eps + np.sqrt(hist_grad[j]))
        weights[j] -= alpha * adj_grad
t1 = time.time()
print('Elapsed time: %.2fs (probably faster than Julia)' % (t1-t0))


# In[ ]:

plot_results(x, y, Z, losses)


# ## Adagrad

# In[ ]:

# training data
x = np.arange(0, 5, 0.1)
y = (x ** 2 + 10*np.sin(x))
x2d = np.concatenate([x, np.ones(x.shape)]).reshape(2, -1).T

# transition function
f, delta_f = get_transition_func("ReLU")

# initialize weights, historical gradients, losses
weights = initweights([2,200,1])
losses = np.zeros(M)

hist_grad = np.copy(weights)
for j in range(len(weights)):
    hist_grad[j] = hist_grad[j] * 0

alpha = 0.02
M = 10000
beta = 0.8
eps = 1e-6

losses = np.zeros(M)
t0 = time.time()
for i in range(M):
    f, delta_f = get_transition_func("ReLU")
    A, Z = forward_pass(weights, x2d, f)
    losses[i] = compute_loss(Z,y)
    gradients = backprop(weights,A,Z,y,delta_f)
    for j in range(len(weights)):
        hist_grad[j] += gradients[j] ** 2
        adj_grad = gradients[j] / (eps + np.sqrt(hist_grad[j]))
        weights[j] -= alpha * adj_grad
t1 = time.time()
print('Elapsed time: %.2fs (probably faster than Julia)' % (t1-t0))


# In[ ]:

plot_results(x, y, Z, losses)

