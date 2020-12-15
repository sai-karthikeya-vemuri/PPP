import autodiff as ad 
import numpy as np 
import matplotlib.pyplot as plt 
from NN_architecture import NeuralNetLSTM
from scipy.io import loadmat
import sys
sys.setrecursionlimit(5000)
def loss_calculator(model,x0,u0,IRK_weights,):
    X=ad.Variable(x0,name="X")
    U = model.output(X)
    U1 = U[:,:-1]
    dummy1 = ad.Variable(np.ones((250,32)),name="dummy1")
    dummy2 =ad.Variable(np.ones((250,33)),name="dummy2")
    dummy3= np.ones((250,32))
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
    temp =0.4*ad.MatMul(F,IRK_weights.transpose())
    U0 = U - temp
    u0 = ad.Variable(u0,name="u0")
    #val = ad.ReduceSumToShape(U0,(250,1))
    lossd = ad.ReduceSumToShape(ad.Pow(U0 - u0,2),(1,1))
    #print(lossd())
    X1 = ad.Variable(np.vstack((-1.0,+1.0)),name="X1")
    Ub = model.output(X1)
    lossb = ad.ReduceSumToShape(ad.Pow(Ub,2),(1,1))
    loss = lossd + lossb

    
    
    return loss


N = 250
data = loadmat('initial_data.mat')
t = data['t'].flatten()[:,None] 
x = data['x'].flatten()[:,None] 
Exact = np.real(data['usol']).T 
    
idx_t0 = 10
idx_t1 = 90
dt = t[idx_t1] - t[idx_t0]
idx_x = np.random.choice(Exact.shape[1], N, replace=False) 
x0 = x[idx_x,:]
u0 = Exact[idx_t0:idx_t0+1,idx_x].T
model = NeuralNetLSTM(20,1,1,33)
tmp = np.float32(np.loadtxt('irk32.txt' , ndmin = 2))
IRK_weights = np.reshape(tmp[0:32**2+32], (32+1,32))
IRK_times = tmp[32**2+32:]
epochs = 5000
optimizer = ad.Adam(len(model.get_weights()))
for i in range(epochs):
    loss = loss_calculator(model,x0,u0,IRK_weights)
    if loss() < 1e-3:
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
    
X = ad.Variable(x0,name="X")
Unew = model.output(X)
U=Unew[:,-1]()
plt.scatter(x0,U)
plt.show()