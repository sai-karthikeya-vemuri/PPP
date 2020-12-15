import autodiff as ad 
import numpy as np 
import matplotlib.pyplot as plt 
from NN_architecture import NeuralNetLSTM,lstm_layer
import matplotlib.pyplot as plt 
import random
import sys
sys.setrecursionlimit(5000)


"""
I hate my life..
just checking whether higher order ode is correct

"""

def loss_domain(model,point):
    point = ad.Variable(np.array([[point]]),name="point")
    
    val = point +  (point*(1-point)*model.output(point))

    du = ad.grad(val,[point])[0]
    d2u = ad.grad(du,[point])[0]
    loss = d2u - point

    return ad.Pow(loss,2)

def sampler(n):
    np.random.seed(0)
    return np.random.uniform(0,3,n) 
model = NeuralNetLSTM(10,1,1,1)

def analytical(x):
    return ((1/6)*x**3) + ((5/6)*x)

resampler_epochs =[0]
for k in resampler_epochs:
    print("sampling for iteration:",k)
    listx= sampler(250)
    epochs = 100

    for j in range(epochs):
        L1 = ad.Variable(0,"L1")
        for i in listx:
            L1.value = L1.value + loss_domain(model,i)()[0][0] 
        L1.value = L1.value /250
        print("initial loss",L1())
        optimizer = ad.Adam(len(model.get_weights()),lr=0.0005)
        for i in listx:
            params = model.get_weights()

            grad_params = ad.grad(loss_domain(model,i),params)
            new_params=[0 for _ in params]
            new_params = optimizer([i() for i in params], [i() for i in grad_params])
            model.set_weights(new_params)

        L2 = ad.Variable(0,"L2")
        for i in listx:
            L2.value = L2.value + loss_domain(model,i)()[0][0]
        L2.value = L2.value/250
        print("Now,loss:",L2())
        if L2() > L1() or L2() < 1e-2 or np.abs(L2()-L1()) < 1e-2:
            print("loss minimized:",L2())
            break
        else:
            print("gradient steptaken epoch:",j)







np.random.seed(0)
x_list = np.random.uniform(0,3,250) 
ylist=[]
y_list =[]
for i in x_list:
    X=ad.Variable(np.array([[i]]),name="X")
    #Xm=ad.Variable(np.array([[1-i]]),name="Xm")
    val =X +  (X*(1-X)*model.output(X))
    y_list.append(val()[0][0])
x= np.linspace(-1,5,100)
y = analytical(x)
plt.scatter(x_list,y_list,marker="+",label="predicted")
plt.plot(x,y,label="analytical")
plt.legend()
plt.show()