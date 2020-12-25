import numpy as np



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
            new_params.append(params[i] - self.lr/(np.sqrt(self.runner[i]+ 1e-8)) * grad_params[i])


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





        