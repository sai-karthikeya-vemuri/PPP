import autodiff as ad 
import numpy as np 
import matplotlib.pyplot as plt 
from NN_architecture import NeuralNetLSTM,lstm_layer
import matplotlib.pyplot as plt 
import random
import sys
sys.setrecursionlimit(5000)
from pyDOE import lhs
"""
And again... i don't know why
May be you aregoing to fail.
Just please be ready to fail..it's not that hard...
Just do it from scratch again, I am fed up changing the old one
"""
def loss_calculator(model,x,t,xi,ui,xb,tb):
    N = len(x)
    Ni = len(xi)
    Nb = len(xb) 
    x= ad.Variable(x,name="x")
    x=ad.Reshape(x,(N,1))
    t=ad.Variable(t,name="t")
    t=ad.Reshape(t,(N,1))
    X = ad.Concat(t,x,1)

    ti = ad.Variable(np.zeros_like(xi),name="ti")
    ti= ad.Reshape(ti,(Ni,1))
    xi = ad.Variable(xi,name="xi")
    xi = ad.Reshape(xi,(Ni,1))
    Xi = ad.Concat(ti,xi,1)
    tb= ad.Variable(tb,name="tb")
    tb= ad.Reshape(tb,(Nb,1))
    xb = ad.Variable(xb,name="xb")
    xb = ad.Reshape(xb,(Nb,1))
    Xb = ad.Concat(tb,xb,1)

    u =model.output(X)
    ux = ad.grad(u,[x])[0]
    ut = ad.grad(u,[t])[0]
    uxx = ad.grad(ux,[x])[0]
    nu= 0.01/np.pi

    f = ut + u*ux - nu*uxx

    lossd = ad.ReduceSumToShape(ad.Pow(f,2),(1,1))/N


    ui = ad.Variable(ui,name="ui")
    ui = ad.Reshape(ui,(Ni,1))

    fi = model.output(Xi) - ui

    lossi = ad.ReduceSumToShape(ad.Pow(fi,2),(1,1))/Ni

    fb = model.output(Xb)

    lossb = ad.ReduceSumToShape(ad.Pow(fb,2),(1,1))/Nb









    



     
    
    
    loss = lossd + lossi + lossb
    return loss
model =NeuralNetLSTM(10,2,2,1)
np.random.seed(0)
x = np.random.uniform(-1.0,1.0,1000)
t = np.random.uniform(0.0,1.0,1000)
#t=np.full_like(x,0.5)
xi = np.linspace(-1.0,1.0,200)
ui = -np.sin(np.pi*xi)
xub = np.full(100,1.0)
xlb= np.full(100,-1.0)
xb=[]
for i,j in zip(xub,xlb):
    xb.append(i)
    xb.append(j)
xb = np.array(xb)
tb = np.linspace(0,1.0,200)


epochs = 100
optimizer = ad.SGDOptimizer(len(model.get_weights()))

for i in range(epochs):
    
    

    loss= loss_calculator(model,x,t,xi,ui,xb,tb)
    if i % 25 ==0:
        print("Epoch:",i)
        print("Loss",loss())
    if loss() < 1e-3:
        break
    params = model.get_weights()

    grad_params = ad.grad(loss,params)
    new_params = [0 for _ in params]
    new_params = optimizer([i() for i in params], [i() for i in grad_params])
    model.set_weights(new_params)
    
    
    if i % 50 ==0:
        x_pred=np.linspace(-1,1,50)
        t_pred=np.full_like(x_pred,0.5)
        x_pred= ad.Variable(x_pred,name="x_pred")
        x_pred=ad.Reshape(x_pred,(50,1))
        t_pred=ad.Variable(t_pred,name="t_pred")
        t_pred=ad.Reshape(t_pred,(50,1))
        X_pred = ad.Concat(t_pred,x_pred,1)

        U_pred= model.output(X_pred)
        x_plot=np.linspace(-1,1,50)
        yplot = U_pred()
        plt.plot(x_plot,yplot,label=f'for iter {i} ')
        
        

plt.plot(xi,ui,label="t=0")
plt.legend()
plt.show()


