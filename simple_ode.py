import autodiff as ad 
import numpy as np
from NN_architecture import NeuralNetLSTM,lstm_layer
import matplotlib.pyplot as plt 

"""
just check whether a small ode is ok..
"""
def loss_domain(model,point):
    point = ad.Variable(np.array([[point]]),name="point")
    val = model.output(point)
    loss =    ad.grad(model.output(point),[point])[0] - ad.Exp(point + 4*model.output(point))

    return ad.Pow(loss,2)
def loss_boundary(model):
    point = ad.Variable(np.array([[0]]),name="point")


    return ad.Pow(model.output(point),2)
def sampler(n):
    return np.random.uniform(-5,-1,n) 
model = NeuralNetLSTM(5,1,1,1)
listx= sampler(500)
epochs = 100

for j in range(epochs):
    L1 = ad.Variable(0,"L1")
    for i in listx:
        L1.value = L1.value + loss_domain(model,i)()[0][0] + loss_boundary(model)()[0][0] 
    print("initial loss",L1())
    optimizer = ad.Adam(len(model.get_weights()),lr=0.01)
    for i in listx:
        params = model.get_weights()

        grad_params = ad.grad(loss_domain(model,i)+loss_boundary(model),params)
        new_params=[0 for _ in params]
        new_params = optimizer([i() for i in params], [i() for i in grad_params])
        model.set_weights(new_params)

    L2 = ad.Variable(0,"L2")
    for i in listx:
        L2.value = L2.value + loss_domain(model,i)()[0][0] + loss_boundary(model)()[0][0] 
    print("Now,loss:",L2())
    if L2() > L1() or L2() < 1e-2 or np.abs(L2()-L1()) < 1e-2:
        print("loss minimized:",L2())
        break
    else:
        print("gradient steptaken epoch:",j)


x_list = np.random.uniform(low=-5,high=-1,size=100)
def y(x):
    return -np.log(5-(4*np.exp(x)))
y_list =[]
for i in x_list:
    X=ad.Variable(np.array([[i]]),name="X")
    y_list.append((model.output(X))()[0][0])
plt.scatter(x_list,y_list)
plt.scatter(x_list,y(x_list))
plt.show()
