"""
This is a validation case solving a simple higher order differential equation.

The differential equation:
y'' -y =0
with BCs
y(0)=0
y(1)=1

Strategy: 
-> the sampler is seeded and plotting is done wider than the trained domain to show that extension over trained boundaries is poor for this method.
-> one training step means one descent step taken by calculating loss at all points one after another.
Type: Hard assignment of BCs. The BCs are embedded by using a trial solution that automatically satisfies the BCs

"""


#Import required packages
import autodiff as ad 
import numpy as np 
import matplotlib.pyplot as plt 
from NN_architecture import NeuralNetLSTM,lstm_layer,xavier
import matplotlib.pyplot as plt 
import random
import sys
from optimizers import *
sys.setrecursionlimit(5000)

#styling of plots
plt.style.use('dark_background')
def loss_domain(model,point):
    """
    Calculates the loss within the domain of the differential equation
    inputs:
    model: The Neural Network model to be trained
    point: The point at which loss should be calculated(should lie within the domain)
    returns: Squared loss in domain
    
    """
    point = ad.Variable(np.array([[point]]),name="point")
    #This is the formulation of trial solution that automatically satisfies the Boundary Conditions
    val = point +  (point*(1-point)*model.output(point))

    du = ad.grad(val,[point])[0]
    d2u = ad.grad(du,[point])[0]
    loss = d2u - point

    return ad.Pow(loss,2)

def sampler(n):
    """
    samples of random data points(uniformly distributed)
    inputs:
    n : number of data points

    returns array of size n  
    """
    np.random.seed(0)
    return np.random.uniform(-1,5,n) 
#Instantiation of Model
model = NeuralNetLSTM(10,1,1,1)
model.set_weights([xavier(i().shape[0],i().shape[1]) for i in model.get_weights()])

def analytical(x):
    """
    Analytical solution to plot 
    inputs:
    x :Array 
    returns array of values corrsponding to the solution of diff eq.
    """
    return ((1/6)*x**3) + ((5/6)*x)

resampler_epochs =[0]
#-------------------------------------------------Start of Training---------------------
for k in resampler_epochs:
    print("sampling for iteration:",k)
    listx= sampler(250)
    epochs = 100
    #Instantiating the optimizer
    optimizer = Adam(len(model.get_weights()))

    for j in range(epochs):
        L1 = ad.Variable(0,"L1")
        for i in listx:
            L1.value = L1.value + loss_domain(model,i)()[0][0] 
        L1.value = L1.value /250
        print("initial loss",L1())
        
        for i in listx:
            params = model.get_weights()
            #Get Gradients

            grad_params = ad.grad(loss_domain(model,i),params)
            new_params=[0 for _ in params]
            new_params = optimizer([i() for i in params], [i() for i in grad_params])
            model.set_weights(new_params)

        L2 = ad.Variable(0,"L2")
        for i in listx:
            L2.value = L2.value + loss_domain(model,i)()[0][0]
        L2.value = L2.value/250
        print("Now,loss:",L2())
        #Exit condition
        if L2() > L1() or L2() < 1e-2 or np.abs(L2()-L1()) < 1e-2:
            print("loss minimized:",L2())
            break
        else:
            print("gradient steptaken epoch:",j)






#-------------------------------------------plotting-------------------------------------
np.random.seed(0)
x_list = np.linspace(-2,6,250) 
ylist=[]
y_list =[]
for i in x_list:
    X=ad.Variable(np.array([[i]]),name="X")
    #Xm=ad.Variable(np.array([[1-i]]),name="Xm")
    val =X +  (X*(1-X)*model.output(X))
    y_list.append(val()[0][0])
x= np.linspace(-2,6,100)
y = analytical(x)
plt.plot(x_list,y_list,marker="+",label="predicted by NN")
plt.plot(x,y,label="analytical solution")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Training NN to solve differential equation: y''-y=0")
plt.legend()
plt.show()