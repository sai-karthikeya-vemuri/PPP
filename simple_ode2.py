"""
This script validates the proposed method by solving a small differential equation whose solution is given by the variable seperable method.

The differential equation:
y' = y
BCs:
y(2) = exp(2)
y(3) = exp(3)

Strategy: 
-> the sampler is seeded and plotting is done only for the trained data points.
-> one training step means one descent step taken by calculating loss at all points one after another.
Type: Soft assignment of Boundary Conditions(deep galerkin approach)

"""
#Import required packages
import autodiff as ad 
import numpy as np
from NN_architecture import NeuralNetLSTM,lstm_layer,xavier
import matplotlib.pyplot as plt 
from optimizers import *
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
    
    val = model.output(point)
    loss =    ad.grad(val,[point])[0] - val
    #print("loss:",loss())
    return ad.Pow(loss,2)
def loss_boundary(model):
    """
    Calculates loss at the boundaries.
    Inputs:
    model: The Neural Network model to be trained
    returns: Sum of Squared loss at the upper and lower boundaries
    """
    #point = ad.Variable(np.array([[0]]),name="point")
    pointu =ad.Variable(np.array([[2]]),name="pointu")
    pointm =ad.Variable(np.array([[3]]),name="pointm")
    #val = model.output(point)-np.array([[1]])
    valu =model.output(pointu)-np.array([[np.exp(2)]])
    valm = model.output(pointm)-np.array([[np.exp(3)]])


    return ad.Pow(valu+valm,2)
def sampler(n):
    """
    samples of random data points(uniformly distributed)
    inputs:
    n : number of data points

    returns array of size n  
    
    """
    np.random.seed(0)
    return np.random.uniform(2,3,n) 
#Instantiating the NN
model = NeuralNetLSTM(10,1,1,1)
model.set_weights([xavier(i().shape[0],i().shape[1]) for i in model.get_weights()])
loss = loss_domain(model,5)
print(loss())
#----------------------------Starting the training------------------------------------------------------------------------
resampler_epochs =[0]
for k in resampler_epochs:
    print("sampling for iteration:",k)
    listx= sampler(250)
    epochs = 100
    #Instantiating the optimizer
    optimizer = Adam(len(model.get_weights()),lr=0.0005)

    for j in range(epochs):
        L1 = ad.Variable(0,"L1")
        for i in listx:
            L1.value = L1.value + loss_domain(model,i)()[0][0] + loss_boundary(model)()[0][0]
        L1.value = L1.value /250
        print("initial loss",L1())
        
        for i in listx:
            params = model.get_weights()
            #Get gradients

            grad_params = ad.grad(loss_domain(model,i)+loss_boundary(model),params)
            new_params=[0 for _ in params]
            new_params = optimizer([i() for i in params], [i() for i in grad_params])
            model.set_weights(new_params)

        L2 = ad.Variable(0,"L2")
        for i in listx:
            L2.value = L2.value + loss_domain(model,i)()[0][0] + loss_boundary(model)()[0][0]
        L2.value = L2.value/250
        print("Now,loss:",L2())
        #Exit condition
        if L2() > L1() or L2() < 1e-2 or np.abs(L2()-L1()) < 1e-2:
            print("loss minimized:",L2())
            break
        else:
            print("gradient steptaken epoch:",j)




#=------------------------------------------------plotting-----------------------------------
np.random.seed(0)
x_list = np.random.uniform(low=1.5,high=3.5,size=250)
plot_list = np.random.uniform(low=1.5,high=3.5,size=500)
def y(x):
    return np.exp(x)
y_list =[]
for i in x_list:
    X=ad.Variable(np.array([[i]]),name="X")
    Xm=ad.Variable(np.array([[1-i]]),name="Xm")
    val =model.output(X) 
    y_list.append(val()[0][0])
plt.scatter(plot_list,y(plot_list),marker="+",label="Analytical")
plt.scatter(x_list,y_list,marker="x",label="Predicted")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Training NN to solve differential equation y'-y=0")
plt.legend()
plt.show()