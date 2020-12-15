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
    loss =    ad.grad(val,[point])[0] - val
    #print("loss:",loss())
    return ad.Pow(loss,2)
def loss_boundary(model):
    #point = ad.Variable(np.array([[0]]),name="point")
    pointu =ad.Variable(np.array([[2]]),name="pointu")
    pointm =ad.Variable(np.array([[3]]),name="pointm")
    #val = model.output(point)-np.array([[1]])
    valu =model.output(pointu)-np.array([[np.exp(2)]])
    valm = model.output(pointm)-np.array([[np.exp(3)]])


    return ad.Pow(valu+valm,2)
def sampler(n):
    np.random.seed(0)
    return np.random.uniform(2,3,n) 
model = NeuralNetLSTM(10,1,1,1)
loss = loss_domain(model,5)
print(loss())

resampler_epochs =[0]
for k in resampler_epochs:
    print("sampling for iteration:",k)
    listx= sampler(250)
    epochs = 100

    for j in range(epochs):
        L1 = ad.Variable(0,"L1")
        for i in listx:
            L1.value = L1.value + loss_domain(model,i)()[0][0] + loss_boundary(model)()[0][0]
        L1.value = L1.value /250
        print("initial loss",L1())
        optimizer = ad.Adam(len(model.get_weights()),lr=0.0005)
        for i in listx:
            params = model.get_weights()

            grad_params = ad.grad(loss_domain(model,i)+loss_boundary(model),params)
            new_params=[0 for _ in params]
            new_params = optimizer([i() for i in params], [i() for i in grad_params])
            model.set_weights(new_params)

        L2 = ad.Variable(0,"L2")
        for i in listx:
            L2.value = L2.value + loss_domain(model,i)()[0][0] + loss_boundary(model)()[0][0]
        L2.value = L2.value/250
        print("Now,loss:",L2())
        if L2() > L1() or L2() < 1e-2 or np.abs(L2()-L1()) < 1e-2:
            print("loss minimized:",L2())
            break
        else:
            print("gradient steptaken epoch:",j)

np.random.seed(0)
x_list = np.random.uniform(low=2,high=3,size=250)
plot_list = np.random.uniform(low=-1,high=4,size=500)
def y(x):
    return np.exp(x)
y_list =[]
for i in x_list:
    X=ad.Variable(np.array([[i]]),name="X")
    Xm=ad.Variable(np.array([[1-i]]),name="Xm")
    val =model.output(X) 
    y_list.append(val()[0][0])
plt.scatter(plot_list,y(plot_list),marker="+",label="Analytical")
plt.scatter(x_list,y_list,marker="x",label="Predicted")

plt.show()