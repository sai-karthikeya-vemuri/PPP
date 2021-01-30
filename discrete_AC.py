"""
Solve time discretized version of Allen-Cahn Equation(which describes the process of phase separation in multi-component alloy systems).

strategy:
->Output of NN is calculated simultaneously for all data points and loss is squared average from all data points.
->The loss is of both domain and boundary together. 
Type:
->Soft assignment of BCs. (explicitly not stated , but trained using data points)
->Since we are using time discretized version , the discretized version(formulated using Runge-Kutta Stepping scheme and solution at the beginning) is solved by the NN.
"""

#Import required packages
import autodiff as ad 
import numpy as np 
import matplotlib.pyplot as plt 
from NN_architecture import NeuralNetLSTM,xavier
from scipy.io import loadmat
import sys
from optimizers import Adam




#Since the setup is complex, recursionlimit should be set to a large value manually
sys.setrecursionlimit(5000)
#styling of plots
plt.style.use('dark_background')
def loss_calculator(model,x0,u0,IRK_weights):
    """
    calculates squared loss at all data points in both domain and boundary .
    Inputs:
    model: The model to be trained 
    x0   : Data points 
    u0   : Solution at 0.1 second 
    IRK Weights : Butcher Tableau of corresponding Runge-Kutta weight matrix
    returns: mean squared loss 
    """
    #converting data points into suitable type
    X=ad.Variable(x0,name="X")
    U = model.output(X)
    U1 = U[:,:-1]
    #Instantiating dummy variables to enable forward gardients
    dummy1 = ad.Variable(np.ones((150,32)),name="dummy1")
    dummy2 =ad.Variable(np.ones((150,33)),name="dummy2")
    dummy3 = ad.Variable(np.ones((2,33)),name="dummy3")
    #Taking gradients and formulating the loss function
    gx = ad.grad(ad.grad(U,[X],dummy2)[0],[dummy2])[0]
    ux = gx[:,:-1]
    gxx= ad.grad(ad.grad(gx,[X],dummy2)[0],[dummy2])[0]
    uxx = gxx[:,:-1]
    F = 5.0*U1 - 5.0*ad.Pow(U1,3) + 0.0001*uxx
    temp =0.4*ad.MatMul(F,IRK_weights.transpose())
    #print(temp.shape)
    U0 = U - temp
    u0 = ad.Variable(u0,name="u0")
    #loss vector on domain
    vald = u0-U0
    #summing over all the axes
    lossd = ad.ReduceSumToShape(ad.Pow(vald,2),())
    
    X1 = ad.Variable(np.vstack((-1.0,+1.0)),name="X1")
    Ub = model.output(X1)
    
    ubx = ad.grad(ad.grad(Ub,[X1],dummy3)[0],[dummy3])[0]
    
    #Loss vector on boundary
    loss_b_val = Ub[0,:] - Ub[1,:]
    #print(loss_b_val.shape)
    lossb = ad.ReduceSumToShape(ad.Pow(loss_b_val,2),())
    #print(lossb())
    #Loss vector on another boundary
    lossbx_val = ubx[0,:] - ubx[1,:]
    lossbx = ad.ReduceSumToShape(ad.Pow(lossbx_val,2),())
    #print(lossbx())
    return lossb + lossbx + lossd


#Number of data points


N = 150
#Taking the weight matrix into a numpy array 
tmp = np.float32(np.loadtxt('irk32.txt' , ndmin = 2))
IRK_weights = np.reshape(tmp[0:32**2+32], (32+1,32))
IRK_times = tmp[32**2+32:]
#Loading the data taken from https://github.com/maziarraissi/PINNs for starter solution and validation of trained model.
data = loadmat('AC.mat')
Exact = np.real(data['uu']).T # T x N
idx_x = np.random.choice(Exact.shape[1], N, replace=False) 
idx_t0 = 20
x = data['x'].flatten()[:,None] # N x 1
x0 = x[idx_x,:]
u0 = Exact[idx_t0:idx_t0+1,idx_x].T
u05 = Exact[60:60+1,idx_x].T


t = data['tt'].flatten()[:,None] # T x 1
#Plot starting solution and actual solution at 0.5s
plt.scatter(x0,u0,label="Starting Solution at t=0.1s",marker=".")
plt.scatter(x0,u05,label="actual at 0.5",marker="+")
#Instantiating the Neural Network
model = NeuralNetLSTM(25,1,1,33)
model.set_weights([xavier(i().shape[0],i().shape[1]) for i in model.get_weights()])

epochs = 5001
#Instantiating the optimizer
optimizer = Adam(len(model.get_weights()))


#--------------------------------------------------------Training-----------------------------------------
for i in range(epochs):
    loss = loss_calculator(model,x0,u0,IRK_weights)
    if i%100==0:
        print("Loss is",loss())
    

    params = model.get_weights()
    #Take gradients
    grad_params = ad.grad(loss,model.get_weights())
    #print("grad params shape",grad_params[6]()) 
    new_params = [0 for _ in params]
    #Calling the optimizer
    new_params = optimizer([i() for i in params], [i() for i in grad_params])
    model.set_weights(new_params)
    #Plot for every 500th iteration
X = ad.Variable(x0,name="X")
Unew = model.output(X)
U=Unew[:,-1]()
plt.scatter(x0,U,label=f'for iter {i} ',marker="x")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Training NN to solve the Allen Cahn Equation")
plt.legend()
plt.savefig(f'AC0.5{i}.png')
    


#plt.legend()
plt.show()