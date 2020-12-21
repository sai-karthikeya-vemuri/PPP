import autodiff as ad 
import numpy as np 
import matplotlib.pyplot as plt 
from NN_architecture import NeuralNetLSTM
from scipy.io import loadmat
import sys
sys.setrecursionlimit(5000)

def loss_calculator(model,x0,u0,IRK_weights):
    
    X=ad.Variable(x0,name="X")
    U = model.output(X)
    U1 = U[:,:-1]
    dummy1 = ad.Variable(np.ones((200,32)),name="dummy1")
    dummy2 =ad.Variable(np.ones((200,33)),name="dummy2")
    dummy3 = ad.Variable(np.ones((2,33)),name="dummy3")
    gx = ad.grad(ad.grad(U,[X],dummy2)[0],[dummy2])[0]
    ux = gx[:,:-1]
    gxx= ad.grad(ad.grad(gx,[X],dummy2)[0],[dummy2])[0]
    uxx = gxx[:,:-1]
    F = 5.0*U1 - 5.0*ad.Pow(U1,3) + 0.0001*uxx
    temp =0.4*ad.MatMul(F,IRK_weights.transpose())
    #print(temp.shape)
    U0 = U - temp
    u0 = ad.Variable(u0,name="u0")
    vald = u0-U0
    lossd = ad.ReduceSumToShape(ad.Pow(vald,2),())
    
    X1 = ad.Variable(np.vstack((-1.0,+1.0)),name="X1")
    Ub = model.output(X1)
    
    ubx = ad.grad(ad.grad(Ub,[X1],dummy3)[0],[dummy3])[0]
    
    loss_b_val = Ub[0,:] - Ub[1,:]
    #print(loss_b_val.shape)
    lossb = ad.ReduceSumToShape(ad.Pow(loss_b_val,2),())
    #print(lossb())
    lossbx_val = ubx[0,:] - ubx[1,:]
    lossbx = ad.ReduceSumToShape(ad.Pow(lossbx_val,2),())
    #print(lossbx())
    return lossb + lossbx + lossd



N = 200
data = loadmat('AC.mat')
t = data['tt'].flatten()[:,None] # T x 1
x = data['x'].flatten()[:,None] # N x 1
Exact = np.real(data['uu']).T # T x N
idx_t0 = 20
idx_x = np.random.choice(Exact.shape[1], N, replace=False) 
x0 = x[idx_x,:]
u0 = Exact[idx_t0:idx_t0+1,idx_x].T
u05 = Exact[60:60+1,idx_x].T
plt.scatter(x0,u0,label="Starting Solution at t=0.1s")
plt.scatter(x0,u05,label="actual at 0.5")
model = NeuralNetLSTM(50,1,1,33)
tmp = np.float32(np.loadtxt('irk32.txt' , ndmin = 2))
IRK_weights = np.reshape(tmp[0:32**2+32], (32+1,32))
IRK_times = tmp[32**2+32:]
epochs = 5001
optimizer = ad.Adam(len(model.get_weights()))


for i in range(epochs):
    loss = loss_calculator(model,x0,u0,IRK_weights)
    if i%100==0:
        print("Loss is",loss())
    

    params = model.get_weights()
    grad_params = ad.grad(loss,model.get_weights())
    #print("grad params shape",grad_params[6]()) 
    new_params = [0 for _ in params]
    new_params = optimizer([i() for i in params], [i() for i in grad_params])
    model.set_weights(new_params)
    
    if i>0 and i % 500==0:
        print("Iteration",i)
        X = ad.Variable(x0,name="X")
        Unew = model.output(X)
        U=Unew[:,-1]()
        plt.scatter(x0,U,label=f'for iter {i} ')
    


plt.legend()
plt.show()