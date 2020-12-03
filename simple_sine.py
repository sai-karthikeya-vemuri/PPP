import autodiff as ad 
import numpy as np
from NN_architecture import NeuralNetLSTM,lstm_layer
import matplotlib.pyplot as plt 

"""
Just a small sine function to check if at all NN is okay ?!

"""

def loss(model,point):
    point = ad.Variable(np.array([[point]]),name="point")
    val = model.output(point)
    loss = val - ad.Sine(point)

    return ad.Pow(loss,2)

def sampler(n):
    return np.random.uniform(0,np.pi,n) 
model = NeuralNetLSTM(5,1,1,1)
listx= sampler(500)
#print(listx)








epochs = 100

for j in range(epochs):
    L1 = ad.Variable(0,"L1")
    for i in listx:
        L1.value = L1.value + loss(model,i)()[0][0]
    print("initial loss",L1())
    for i in listx:
        params = model.get_weights()

        grad_params = ad.grad(loss(model,i),params)
        new_params=[0 for _ in params]
        for i in range(len(params)):

            new_params[i]=params[i]()- 0.0146*grad_params[i]()
        model.set_weights(new_params)

    L2 = ad.Variable(0,"L2")
    for i in listx:
        L2.value = L2.value + loss(model,i)()[0][0]
    print("Now,loss:",L2())
    if L2() > L1() or L2() < 1e-2 or np.abs(L2()-L1()) < 1e-2:
        print("loss minimized:",L2())
        break
    else:
        print("gradient steptaken epoch:",j)


x_list = np.random.uniform(low=0,high=2*np.pi,size=100)
def y(x):
    return np.sin(x)
y_list =[]
for i in x_list:
    X=ad.Variable(np.array([[i]]),name="X")
    y_list.append(model.output(X)()[0][0])
plt.scatter(x_list,y_list)
plt.scatter(x_list,y(x_list))
plt.show()