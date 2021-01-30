"""
Solve time discretized version of Burgers Equation ( suitable for analysis in various fields like sedimentation of polydispersive suspensions and colloids, aspect of turbulence, non-linear wave
propagation, growth of molecular interfaces, longitudinal elastic waves in isotropic solids, traffic flow, cosmology, gas dynamics and shock wave theory.).

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
from optimizers import Adam
import sys
#styling of plots
plt.style.use('dark_background')
#Since the setup is complex, recursionlimit should be set to a large value manually
sys.setrecursionlimit(10000)
def loss_calculator(model,x0,u0,IRK_weights,):
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
    #Instantiating dummy variables to get forward gradients
    dummy1 = ad.Variable(np.ones((100,32)),name="dummy1")
    dummy2 =ad.Variable(np.ones((100,33)),name="dummy2")
    dummy3= np.ones((100,32))
    #g = ad.grad(U,[X],previous_grad=dummy2)[0]
    #print("g:",g().shape)
    gx = ad.grad(ad.grad(U,[X],dummy2)[0],[dummy2])[0]
    ux = gx[:,:-1]
    #print("ux",ux().shape)
    #g1 = ad.grad(g,[X],previous_grad=dummy1)[0]
    #print("g1",g1().shape)
    gxx= ad.grad(ad.grad(gx,[X],dummy2)[0],[dummy2])[0]
    uxx = gxx[:,:-1]
    
    #print("uxx",uxx().shape)
    #F = -U1*g + (0.01/np.pi)*g1
    F = -U1*ux + ((0.01/np.pi)*uxx)
    #Formulate the loss
    temp =0.4*ad.MatMul(F,IRK_weights.transpose())
    U0 = U - temp
    u0 = ad.Variable(u0,name="u0")
    #val = ad.ReduceSumToShape(U0,(250,1))
    #squared Sum over all axes
    lossd = ad.ReduceSumToShape(ad.Pow(U0 - u0,2),(1,1))
    #print(lossd())
    X1 = ad.Variable(np.vstack((-1.0,+1.0)),name="X1")
    Ub = model.output(X1)
    #Loss at boundary squared sum over all axes
    lossb = ad.ReduceSumToShape(ad.Pow(Ub,2),(1,1))
    loss = lossd + lossb

    
    
    return loss

#Loading of exact data for plotting and training
N = 100
data = loadmat('initial_data.mat')
Exact = np.real(data['usol']).T 
idx_x = np.random.choice(Exact.shape[1], N, replace=False) 
t = data['t'].flatten()[:,None] 


    
idx_t0 = 10
u0 = Exact[idx_t0:idx_t0+1,idx_x].T
x = data['x'].flatten()[:,None] 

x0 = x[idx_x,:]
#Instantiating the NN
model = NeuralNetLSTM(20,1,1,33)
model.set_weights([xavier(i().shape[0],i().shape[1]) for i in model.get_weights()])
tmp = np.float32(np.loadtxt('irk32.txt' , ndmin = 2))
IRK_weights = np.reshape(tmp[0:32**2+32], (32+1,32))
IRK_times = tmp[32**2+32:]
epochs = 5001
#Instantiating the optimizer
optimizer = Adam(len(model.get_weights()))
#------------------------------------------------------Start of training-------------------------------------
for i in range(epochs):
    loss = loss_calculator(model,x0,u0,IRK_weights)
    #Exit condition
    if loss() <= 1:
        break
    if i % 50 ==0:
        print("Iteration",i)
        print("loss  epoch",loss())
    params = model.get_weights()
    grad_params = ad.grad(loss,model.get_weights())
    #print("grad params shape",grad_params[6]()) 
    new_params = [0 for _ in params]
    new_params = optimizer([i() for i in params], [i() for i in grad_params])
    model.set_weights(new_params)
#-------------------------------------------plotting-----------------------    
def f(x):
    """
    plots starter at t=0s
    inputs
    x : input array
    return solution at 0s
    """
    return -np.sin(np.pi*x)
temp = np.linspace(-1,1,100)
plt.plot(temp,f(temp),label="Initial Condition at t=0s")
ue = Exact[50:50+1,idx_x].T
plt.scatter(x0,ue,marker="x",label="Exact Solution at t=0.5s")
X = ad.Variable(np.sort(x0),name="X")
Unew = model.output(X)
U=Unew[:,-1]()

plt.scatter(np.sort(x0),U,marker="+",label="Predicted by NN t=0.5s")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Training a NN to solve Burgers Equation")
plt.legend()
plt.show()