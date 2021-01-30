"""
This is a comparision between the optimizers based on loss vs iterations

A simple loss function is defined commonly for all the optimizers .
The same Neural Network is instantiated individually for every optimizer and training is done for 1000 iterations.
Each optimizer object is created and loss is minimized for 50 data points 

"""




#Importing required packages and functions
import numpy as np 
import autodiff as ad
from NN_architecture_2 import *
from optimizers import *


plt.style.use('dark_background')



def loss_calc(model):
    """
    Loss calculator function 
    Input:
    model: The Neural Network object 
    returns total loss calculated at 50 data points 
    """
    def f(x):
        """
        The function against which loss is calculated
        inputs:
        x : number or an array
        return sine of given array x
        """
        return np.sin(x)+np.cos(x) +x 
    x= np.linspace(-np.pi,np.pi,50)
    y = f(x)



    #instantiating the variable and reshaping accordingly
    x= ad.Variable(x,"x")
    x= ad.Reshape(x,(50,1))
    #Predicted output by Neural network
    y_pred = model.output(x)
    #Vector of losses at data points
    f = y_pred - y
    #Sum of squared loss at all data points
    loss = ad.ReduceSumToShape(ad.Pow(f,2),())
    return loss

#Instantiating the Neural Network
model = NeuralNetLSTM(5,0,1,1)

#Instantiating the optimizer
optimizer=Adamax(len(model.get_weights()))
loss_list =[]
#training for 1000 iterations
for i in range(1000):
    params = model.get_weights()
    loss = loss_calc(model)
    print("iteration ",i)
    loss_list.append(loss())
    grad_params = ad.grad(loss,params)
    new_params = optimizer([i() for i in params], [i() for i in grad_params])
    #print(new_params)
    model.set_weights(new_params)


x= np.linspace(0,1000,1000)
plt.plot(x,loss_list,label="Adamax: lr=0.00146,b1=0.9,b2=0.99")




#Instantiating the Neural Network
model = NeuralNetLSTM(5,0,1,1)

#Instantiating the optimizer
optimizer=SGD(lr=1e-6)
loss_list =[]
#training for 1000 iterations
for i in range(1000):
    params = model.get_weights()
    loss = loss_calc(model)
    print("iteration ",i)
    loss_list.append(loss())
    grad_params = ad.grad(loss,params)
    new_params = optimizer([i() for i in params], [i() for i in grad_params])
    #print(new_params)
    model.set_weights(new_params)


x= np.linspace(0,1000,1000)
plt.plot(x,loss_list,label="SGD:lr=1e-6")


#Instantiating the Neural Network
model = NeuralNetLSTM(5,0,1,1)

#Instantiating the optimizer
optimizer=Momentum(len(model.get_weights()),lr=1e-6)
loss_list =[]
#training for 1000 iterations
for i in range(1000):

    params = model.get_weights()
    loss = loss_calc(model)
    print("iteration ",i)
    loss_list.append(loss())
    grad_params = ad.grad(loss,params)
    new_params = optimizer([i() for i in params], [i() for i in grad_params])
    #print(new_params)
    model.set_weights(new_params)

x= np.linspace(0,1000,1000)
plt.plot(x,loss_list,label="Momenta: lr=1e-6,gamma=0.9")


#Instantiating the Neural Network
model = NeuralNetLSTM(5,0,1,1)

#Instantiating the optimizer
optimizer=Adagrad(len(model.get_weights()))
loss_list =[]
#training for 1000 iterations
for i in range(1000):
    params = model.get_weights()
    loss = loss_calc(model)
    print("iteration ",i)
    loss_list.append(loss())
    grad_params = ad.grad(loss,params)
    new_params = optimizer([i() for i in params], [i() for i in grad_params])
    #print(new_params)
    model.set_weights(new_params)

x= np.linspace(0,1000,1000)
plt.plot(x,loss_list,label="Adagrad:lr=0.00146")



#Instantiating the Neural Network
model = NeuralNetLSTM(5,0,1,1)

#Instantiating the optimizer
optimizer=RMSProp(len(model.get_weights()))
loss_list =[]
#training for 1000 iterations
for i in range(1000):
    params = model.get_weights()
    loss = loss_calc(model)
    print("iteration ",i)
    loss_list.append(loss())
    grad_params = ad.grad(loss,params)
    new_params = optimizer([i() for i in params], [i() for i in grad_params])
    #print(new_params)
    model.set_weights(new_params)

x= np.linspace(0,1000,1000)
plt.plot(x,loss_list,label="RMSProp:lr=0.00146,decay_rate=0.9")


#Instantiating the Neural Network
model = NeuralNetLSTM(5,0,1,1)

#Instantiating the optimizer
optimizer=Adam(len(model.get_weights()))
loss_list =[]
#training for 1000 iterations
for i in range(1000):
    params = model.get_weights()
    loss = loss_calc(model)
    print("iteration ",i)
    loss_list.append(loss())
    grad_params = ad.grad(loss,params)
    new_params = optimizer([i() for i in params], [i() for i in grad_params])
    #print(new_params)
    model.set_weights(new_params)

x= np.linspace(0,1000,1000)
plt.plot(x,loss_list,label="Adam:lr=0.00146,b1=0.9,b2=0.999")

plt.xlabel("Iterations",fontsize=10)
plt.ylabel("Loss",fontsize=10)
plt.title("Loss vs Iterations",fontsize=15)

plt.legend()
plt.show()



        