import autodiff as ad 
import numpy as np 
import matplotlib.pyplot as plt 
from NN_architecture_2 import NeuralNetLSTM,lstm_layer
import matplotlib.pyplot as plt 
import random
import sys
sys.setrecursionlimit(5000)
from pyDOE import lhs
from scipy.io import loadmat

"""
And again... i don't know why
May be you aregoing to fail.
Just please be ready to fail..it's not that hard...
Just do it from scratch again, I am fed up changing the old one
"""
def loss_calculator(model,X_u,X_f,lb,ub,U,N_f,N_u):
    xb =ad.Variable(X_u[:,0:1],"xb")
    tb = ad.Variable(X_u[:,1:2],"tb")

    xd =ad.Variable(X_f[:,0:1],"xd")
    td =  ad.Variable(X_f[:,1:2],"td")
    nu= 0.01/np.pi
    U = ad.Variable(U,"U")


    X_d = ad.Concat(xd,td,1)
    X_b = ad.Concat(xb,tb,1)
    u = model.output(X_d)
    u_t = ad.grad(u,[td])[0]
    u_x = ad.grad(u,[xd])[0]
    u_xx = ad.grad(u_x,[xd])[0]
    f = u_t + u*u_x - nu*u_xx
    lossd = ad.ReduceSumToShape(ad.Pow(f,2),())/N_f

    ub = model.output(X_b)
    fb = ub - U

    lossb = ad.ReduceSumToShape(ad.Pow(fb,2),())/N_u

    loss = lossd + lossb







    return loss



N_f = 2000
N_u = 20
data = loadmat('initial_data.mat')
    
t = data['t'].flatten()[:,None]
x = data['x'].flatten()[:,None]
Exact = np.real(data['usol']).T
    
X, T = np.meshgrid(x,t)
    
X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
u_star = Exact.flatten()[:,None]              


lb = X_star.min(0)
ub = X_star.max(0)    
        
xx1 = np.hstack((X[0:1,:].T, T[0:1,:].T))
uu1 = Exact[0:1,:].T
xx2 = np.hstack((X[:,0:1], T[:,0:1]))
uu2 = Exact[:,0:1]
xx3 = np.hstack((X[:,-1:], T[:,-1:]))
uu3 = Exact[:,-1:]
    
X_u_train = np.vstack([xx1, xx2, xx3])
X_f_train = lb + (ub-lb)*lhs(2, N_f)
X_f_train = np.vstack((X_f_train, X_u_train))
u_train = np.vstack([uu1, uu2, uu3])
    
idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)
X_u_train = X_u_train[idx, :]
u_train = u_train[idx,:]
model = NeuralNetLSTM(20,2,2,1)


epochs = 501
optimizer = ad.Adam(len(model.get_weights()))
for i in range(epochs):

    
    

    loss = loss_calculator(model,X_u_train,X_f_train,lb,ub,u_train,N_f,N_u)
    if i%10 ==0:

        print(loss())
    
    params = model.get_weights()
    grad_params = ad.grad(loss,model.get_weights())
    #print("grad params shape",grad_params[6]()) 
    new_params = [0 for _ in params]
    new_params = optimizer([i() for i in params], [i() for i in grad_params])
    model.set_weights(new_params)

    if i % 100 == 0:
        x_plot = ad.Variable(np.random.uniform(0,1,(100,1)),"x_plot")
        t_plot = ad.Variable(np.full((100,1),0.5),"t_plot")
        X_plot = ad.Concat(x_plot,t_plot,1)

        U_plot = model.output(X_plot)
        plt.scatter(x_plot(),U_plot(),label=f'for iter {i}')
plt.legend()
plt.show()