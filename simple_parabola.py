"""
Validation case:
This script is a validation case for Neural Network Approximation theorem for PDEs,which states that Neural Networks can be used as approximators for functions.
This script approximates a given NN to behave as a gradient of cubic function x**3/Solution to Ode y'=6x , c=0 within trained region.
Strategy:
Calcuating loss and taking descent step for all the data points one after the other constitutes one iteration of training. 
Type: No use of Boundary Conditions
"""


#Import required packages

import autodiff as ad 
import numpy as np
from NN_architecture import NeuralNetLSTM,lstm_layer,xavier
import matplotlib.pyplot as plt 
from optimizers import *

#styling of plots
plt.style.use('dark_background')
def loss(model,point):
    """
    Calculates loss of a model at the given point 
    Inputs:
    model: The NN model which is being trained .Type: Neural Net LSTM Object. 
    point: The point at which the loss is to be calculated. Type:Float
    returns: squared loss of model at the point
    """
    #Conversion of the point into autodiff variable
    point = ad.Variable(np.array([[point]]),name="point")
    val = model.output(point)
    #penalizing against gradient of x**3
    loss = val - (ad.grad(ad.Pow(point,3),[point])[0])

    return ad.Pow(loss,2)

def sampler(n):
    """
    samples of random data points(uniformly distributed)
    inputs:
    n : number of data points

    returns array of size n  
    """
    return np.random.uniform(-5,5,n) 
#Instantiating the Neural Network
model = NeuralNetLSTM(5,1,1,1)
model.set_weights([xavier(i().shape[0],i().shape[1]) for i in model.get_weights()])
listx= sampler(500)
#Max number of iterations
epochs = 150
#-------------------------------------------------------Starting of training---------------------------------


for j in range(epochs):
    L1 = ad.Variable(0,"L1")
    for i in listx:
        L1.value = L1.value + loss(model,i)()[0][0]
    print("initial loss",L1())
    #Instantiating optimizer
    optimizer = RMSProp(len(model.get_weights()))
    for i in listx:
        params = model.get_weights()
        #Get Gradients
        grad_params = ad.grad(loss(model,i),params)
        new_params=[0 for _ in params]
        #Take descent step by calling optimizer
        new_params = optimizer([i() for i in params], [i() for i in grad_params])
        model.set_weights(new_params)

    L2 = ad.Variable(0,"L2")
    for i in listx:
        L2.value = L2.value + loss(model,i)()[0][0]
    print("Now,loss:",L2())
    #Exit condition
    if L2() > L1() or L2() < 1e-2 or np.abs(L2()-L1()) < 1e-2:
        print("loss minimized:",L2())
        break
    else:
        print("gradient steptaken epoch:",j)

#-----------------------------------------------------plotting-----------------------------------------------------------------
x_list = np.linspace(-5,5,100)
def y(x):
    """
    parabola function for plotting 
    input:
    x : float or numpy array of floats
    returns 3*x**2
    """
    return 3*x**2
y_list =[]
for i in x_list:
    X=ad.Variable(np.array([[i]]),name="X")
    y_list.append(model.output(X)()[0][0])
plt.plot(x_list,y_list,marker="_",label="Predicted by NN")
plt.plot(x_list,y(x_list),marker="_",label="Theoretical Solution")
plt.title("Training Neural Network as a functional approximator for parabola :y = 3*x**2")
plt.xlabel("x----------->")
plt.ylabel("y")
plt.legend()
plt.show()
