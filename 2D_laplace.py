#Import required packages
import autodiff as ad 
import numpy as np
from numpy import array
from NN_architecture import NeuralNetLSTM,xavier,diff_n_times
import matplotlib.pyplot as plt 
from optimizers import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
#styling of plots
plt.style.use('dark_background')
def loss_calculator(model,points):
    
    
    points = ad.Variable(points,"points")
    val = model.output(points)
    dp2 = diff_n_times(val,points,2)
    #print(dp2.shape)
    #print([i() for i in dp2])
    f = ad.ReduceSumToShape(dp2,(150,1))
    #print(f.shape)
    lossd = ad.ReduceSumToShape(ad.Pow(f,2),())
    pointsb = ad.Variable(np.array([[0,0],[0,0.25],[0,0.5],[0,0.75],[0,1],[0.25,0],[0.5,0],[0.75,0],[1,0]]))
    fb = model.output(pointsb)
    lossb = ad.ReduceSumToShape(ad.Pow(fb,2),())
    pointsdbx = ad.Variable(np.array([[1,0],[1,0.25],[1,0.5],[1,0.75],[1,1]]))
    #print(pointsdbx[:,0])
    fbx = ad.grad(model.output(pointsdbx),[pointsdbx[:,0]])[0]
    lossbx = ad.ReduceSumToShape(ad.Pow(fbx,2),())
    pointsby=ad.Variable(np.array([[0,1],[0.25,1],[0.5,1],[0.75,1],[1,1]]))
    #print(ad.Sine(1.5*np.pi*pointsby[:,0])())
    
    fby = ad.Reshape(model.output(pointsby),(5,)) - ad.Sine(1.5*np.pi*pointsby[:,0])
    #print(fby.shape)
    lossby = ad.ReduceSumToShape(ad.Pow(fby,2),())
    return lossd + lossb + lossbx + lossby
    


model = NeuralNetLSTM(10,2,2,1)
model.set_weights([xavier(i().shape[0],i().shape[1]) for i in model.get_weights()])
np.random.seed(0)
points= np.random.uniform(0,1,(150,2))
optimizer = Adam(len(model.get_weights()))
epochs =2000
#-------------------------------------------------------Training--------------------------------------------------
for i in range(epochs):
    print("iter",i)
    loss = loss_calculator(model,points)
    print("loss",loss())
    params = model.get_weights()
    grad_params = ad.grad(loss,params)
    new_params = optimizer([i() for i in params], [i() for i in grad_params])
    model.set_weights(new_params)
    loss2= loss_calculator(model,points)
    print("loss now",loss2())
    #Exit condition
    if loss2()< 1e-2:
        break





def z_analytical(x,y):
    X, Y = np.meshgrid(x,y)
    z = np.sinh(1.5*np.pi*Y / x[-1]) /\
    (np.sinh(1.5*np.pi*y[-1]/x[-1]))*np.sin(1.5*np.pi*X/x[-1])
    return z 
x = np.linspace(0,1,100)
y = np.linspace(0,1,100)
z = z_analytical(x,y)
fig = plt.figure(figsize=(11,7), dpi=100)
ax = fig.gca(projection='3d')
X,Y = np.meshgrid(x,y)
z_pred=model.output(points)()
surf = ax.plot_surface(X,Y,z[:], rstride=1, cstride=1, cmap=cm.viridis,linewidth=0, antialiased=False,alpha=0.5,label="Analytical Solution")
surf._facecolors2d=surf._facecolors3d
surf._edgecolors2d=surf._edgecolors3d
ax.scatter(points[:,0],points[:,1],z_pred,color='red',marker='x',label="Predicted by NN") 
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$z$')
ax.set_title("Training of NN to solve 2D Laplace equation(training for 2000 iterations)")
ax.legend()
fig.colorbar(surf, shrink=0.5, aspect=5)
#fig.legend()
#fig.title("Training NN to solve Laplace Equation 2D")
#ax.view_init(30,45)
