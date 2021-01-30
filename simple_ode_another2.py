"""
This script validates the proposed method by solving a small differential equation whose solution is given by the variable seperable method.

The differential equation:
y' = y
BCs:
y(2) = exp(2)
y(3) = exp(3)
strategy:
-> Soft boundary condition assignments
-> loss is calculated for all the points and reduces to its mean square value before taking gradient steps
"""
#Import required packages
import autodiff as ad 
import numpy as np
from numpy import array
from another_NN import NeuralNetLSTM,xavier
import matplotlib.pyplot as plt 
from optimizers import *
#styling of plots
plt.style.use('dark_background')



def loss_calculator(model,points):
    """
    Calculates the loss within the domain nd boundary of the differential equation
    inputs:
    model: The Neural Network model to be trained
    points: The points at which loss should be calculated(should lie within the domain)
    returns:Mean Squared loss from all the points in domain
    """
    X = ad.Variable(points,"X")
    
    val = model.output(X)
    f = ad.grad(val,[X])[0] - val
    lossd = ad.ReduceSumToShape(ad.Pow(f,2),())/100
    fb =model.output(np.array([[2],[3]])) - np.array([[np.exp(2)],[np.exp(3)]])
    lossb = ad.ReduceSumToShape(ad.Pow(fb,2),())
    loss = lossb + lossd
    return loss

def sampler(n):
    """
    samples of random data points(uniformly distributed)
    inputs:
    n : number of data points

    returns array of size n  
    
    """
    np.random.seed(0)
    return np.reshape(np.random.uniform(2,3,n),(n,1)) 

#x=ad.Variable(sampler(100),"x")
#Instantiating model and optimizer
model = NeuralNetLSTM(10,1,1,1)
model1=NeuralNetLSTM(10,1,1,1)
print([i() for i in model.get_weights()])

model.set_weights([xavier(i().shape[0],i().shape[1]) for i in model.get_weights()])
optimizer= RMSProp(len(model.get_weights()))
epochs =1000
x=sampler(100)
#-------------------------------------------------------Training--------------------------------------------------
for i in range(epochs):
    loss = loss_calculator(model,x)
    print("loss",loss())
    params = model.get_weights()
    grad_params = ad.grad(loss,params)
    new_params = optimizer([i() for i in params], [i() for i in grad_params])
    model.set_weights(new_params)
    loss2= loss_calculator(model,x)
    print("loss now",loss2())
    #Exit condition
    if loss2()< 1e-2:
        break
#-----------------------------------Plotting--------------------------------------
x_list = np.random.uniform(low=1.5,high=3.5,size=250)
plot_list = np.random.uniform(low=1.5,high=3.5,size=500)
def y(x):
    return np.exp(x)
y_list =[]
for i in x_list:
    X=ad.Variable(np.array([[i]]),name="X")
    val =model.output(X) 
    y_list.append(val()[0][0])
plt.scatter(plot_list,y(plot_list),marker="+",label="Analytical")
plt.scatter(x_list,y_list,marker="x",label="Predicted")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Training(Mean error reduction) NN to solve differential equation y'-y=0")
plt.legend()
plt.show()

