import numpy as np
import autodiff as ad
from NN_architecture_2 import NeuralNetLSTM
import matplotlib.pyplot as plt 

class optimizer():
    def _forward_pass(self):
        raise NotImplementedError
    def __call__(self,params,grad_params):
        new_params = self._forward_pass(params,grad_params)
        return new_params




class SGD(optimizer):
    def __init__(self,lr=0.00146):
        self.lr = lr
    def _forward_pass(self,params,grad_params):
        if isinstance(params,int):
            n=1
        else:
            n = len(params)
        new_params= []
        for i in range(n):
            new_params.append(params[i] - self.lr*grad_params[i])
        return new_params


class Momentum(optimizer):
    def __init__(self,num_params,lr=0.00146,gamma=0.8):
        self.lr = lr
        self.num_params = num_params
        self.gamma = gamma
        self.num_params = num_params
        self.a = [0 for _ in range(num_params)]

    def _forward_pass(self,params,grad_params):
        new_params=[]
        for i in range(self.num_params):
            self.a[i] = self.gamma * self.a[i] + grad_params[i]
            new_params.append(params[i]- self.lr * self.a[i])
        return new_params
    

class Adagrad(optimizer):
    def __init__(self,num_params,lr=0.00146):
        self.lr = lr
        self.num_params = num_params
        self.runner = [0 for _ in range(num_params)]
    
    def _forward_pass(self,params,grad_params):
        new_params = []
        for i in range(self.num_params):
            self.runner[i] =  self.runner[i]  + grad_params[i]**2
            new_params.append(params[i] - self.lr/(np.sqrt(self.runner[i])+ 1e-8) * grad_params[i])


        return new_params


class Adam(optimizer):
    def __init__(self,num_params,lr=0.00146,b1=0.9,b2=0.999):
        self.counter =0
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.num_params = num_params
        self.momentum = [0 for _ in range(num_params)]
        self.velocity = [0 for _ in range(num_params)]
    def _forward_pass(self,params,grad_params):
        new_params = []
        for i in range(self.num_params):
            self.momentum[i] = self.b1 * self.momentum[i]  + (1-self.b1)*grad_params[i]
            self.velocity[i] = self.b2 * self.velocity[i]  + (1-self.b2)*grad_params[i]**2
            accumulation = self.lr * np.sqrt(1 - self.b2 ** self.counter) / (1 - self.b1 ** self.counter + 1e-8)
            new_params.append(params[i]-accumulation * self.momentum[i] / (np.sqrt(self.velocity[i]) + 1e-8))
        
        self.counter += 1 
        return new_params



class RMSProp(optimizer):
    def __init__(self,num_params,lr=0.00146,decay_rate=0.9):
        self.num_params = num_params
        self.lr = lr
        self.decay_rate = decay_rate
        self.runner = [0 for _ in range(num_params)]
    def _forward_pass(self,params,grad_params):
        new_params = []
        for i in range(self.num_params):
            self.runner[i] =  self.decay_rate*self.runner[i]  + (1-self.decay_rate)*grad_params[i]**2
            new_params.append(params[i] - self.lr/(np.sqrt(self.runner[i])+ 1e-8) * grad_params[i])
        return new_params


class Adamax(optimizer):
    def __init__(self,num_params,lr=0.00146,b1=0.9,b2=0.99):
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.num_params = num_params
        self.counter =0
        self.momentum = [0 for _ in range(num_params)]
        self.velocity = [0 for _ in range(num_params)]
    def _forward_pass(self,params,grad_params):
        self.counter += 1
        epsilon = 1e-8

        new_params=[]
        for i in range(self.num_params):
            self.momentum[i] = self.b1 * self.momentum[i]  + (1-self.b1)*grad_params[i]
            #print(self.momentum[i])
            self.velocity[i] = max(self.b2*self.velocity[i], abs(np.linalg.norm(grad_params[i])))

            
            #print(self.velocity[i])
            mhat = self.momentum[i]/(1-self.b1**self.counter)
            new_params.append(params[i] - (self.lr*mhat)/(self.velocity[i]+epsilon))
        return new_params
        


def loss_calc(model):
    def f(x):
        return np.sin(x)
    x= np.linspace(-1,+1,50)
    y = f(x)


    x= ad.Variable(x,"x")
    x= ad.Reshape(x,(50,1))
    y_pred = model.output(x)
    f = y_pred - y
    loss = ad.ReduceSumToShape(ad.Pow(f,2),())
    return loss


model = NeuralNetLSTM(5,0,1,1)


optimizer=Adamax(len(model.get_weights()))
loss_list =[]
for i in range(1000):
    params = model.get_weights()
    loss = loss_calc(model)
    print("iteration ",i)
    loss_list.append(loss())
    grad_params = ad.grad(loss,params)
    new_params = optimizer([i() for i in params], [i() for i in grad_params])
    #print(new_params)
    model.set_weights(new_params)


x= np.linspace(0,1000,1000)
plt.plot(x,loss_list,label="Adamax")





model = NeuralNetLSTM(5,0,1,1)


optimizer=SGD(lr=1e-5)
loss_list =[]
for i in range(1000):
    params = model.get_weights()
    loss = loss_calc(model)
    print("iteration ",i)
    loss_list.append(loss())
    grad_params = ad.grad(loss,params)
    new_params = optimizer([i() for i in params], [i() for i in grad_params])
    #print(new_params)
    model.set_weights(new_params)


x= np.linspace(0,1000,1000)
plt.plot(x,loss_list,label="SGD")



model = NeuralNetLSTM(5,0,1,1)


optimizer=Momentum(len(model.get_weights()),lr=1e-5)
loss_list =[]
for i in range(1000):
    params = model.get_weights()
    loss = loss_calc(model)
    print("iteration ",i)
    loss_list.append(loss())
    grad_params = ad.grad(loss,params)
    new_params = optimizer([i() for i in params], [i() for i in grad_params])
    #print(new_params)
    model.set_weights(new_params)

x= np.linspace(0,1000,1000)
plt.plot(x,loss_list,label="Momenta")



model = NeuralNetLSTM(5,0,1,1)


optimizer=Adagrad(len(model.get_weights()))
loss_list =[]
for i in range(1000):
    params = model.get_weights()
    loss = loss_calc(model)
    print("iteration ",i)
    loss_list.append(loss())
    grad_params = ad.grad(loss,params)
    new_params = optimizer([i() for i in params], [i() for i in grad_params])
    #print(new_params)
    model.set_weights(new_params)

x= np.linspace(0,1000,1000)
plt.plot(x,loss_list,label="Adagrad")




model = NeuralNetLSTM(5,0,1,1)


optimizer=RMSProp(len(model.get_weights()))
loss_list =[]
for i in range(1000):
    params = model.get_weights()
    loss = loss_calc(model)
    print("iteration ",i)
    loss_list.append(loss())
    grad_params = ad.grad(loss,params)
    new_params = optimizer([i() for i in params], [i() for i in grad_params])
    #print(new_params)
    model.set_weights(new_params)

x= np.linspace(0,1000,1000)
plt.plot(x,loss_list,label="RMSProp")



model = NeuralNetLSTM(5,0,1,1)


optimizer=Adam(len(model.get_weights()))
loss_list =[]
for i in range(1000):
    params = model.get_weights()
    loss = loss_calc(model)
    print("iteration ",i)
    loss_list.append(loss())
    grad_params = ad.grad(loss,params)
    new_params = optimizer([i() for i in params], [i() for i in grad_params])
    #print(new_params)
    model.set_weights(new_params)

x= np.linspace(0,1000,1000)
plt.plot(x,loss_list,label="Adam")
plt.legend()
plt.show()



        