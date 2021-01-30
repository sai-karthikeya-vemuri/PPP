"""
I am going to solve the burgers equation here using the deep NN
and at present thr autodiff package by Bruno Gavranovic'

"""
import numpy as np 
import autodiff as ad 
import os 
import os
#os.environ[“PATH”] += os.pathsep + r'C:\users\91994\appdata\local\programs\python\python37\lib\site-packages\graphviz-2.38\release\bin’

def diff_n_times(graph, wrt, n):
    for i in range(n):
        graph = ad.grad(graph, [wrt])[0]

    return graph
"""
def xavier(input_dim,output_dim):
    r = 4*np.sqrt(6/(input_dim+output_dim))
    
    def random_generator(low,high,input_dim,output_dim):
        return np.random.uniform(low=low,high=high,size=(input_dim,output_dim))
    value = random_generator(-r,+r,input_dim,output_dim)


    return np.array(value)

"""
def xavier(input_dim,output_dim):
    stddev = np.sqrt(2/(input_dim+output_dim))
    return np.random.normal(loc=0.0,scale=stddev,size=(input_dim,output_dim))



class lstm_layer():

    def __init__(self,input_dim,output_dim):
        #self.inputs = inputs
        self.input_dim = input_dim
        self.output_dim= output_dim
        self._Uz = ad.Variable(xavier(self.input_dim,self.output_dim),name="Uz")
        self._Ug = ad.Variable(xavier(self.input_dim,self.output_dim),name="Ug")
        self._Ur = ad.Variable(xavier(self.input_dim,self.output_dim),name="Ur")
        self._Uh = ad.Variable(xavier(self.input_dim,self.output_dim),name="Uh")
        self._Wz = ad.Variable(xavier(self.output_dim,self.output_dim),name="Wz")
        self._Wg = ad.Variable(xavier(self.output_dim,self.output_dim),name="Wg")
        self._Wr = ad.Variable(xavier(self.output_dim,self.output_dim),name="Wr")
        self._Wh = ad.Variable(xavier(self.output_dim,self.output_dim),name="Wh")
        self._bz = ad.Variable(xavier(1,self.output_dim),name = "bz")
        self._bg = ad.Variable(xavier(1,self.output_dim),name = "bg")
        self._br = ad.Variable(xavier(1,self.output_dim),name = "br")
        self._bh = ad.Variable(xavier(1,self.output_dim),name = "bh")
    def output_layer(self,S_Old,X):
        S=S_Old
        val_z = ad.MatMul(X,self._Uz) + ad.MatMul(S,self._Wz) + self._bz
        Z=ad.Sigmoid(val_z)
        #print("Z",Z())
        val_g = ad.MatMul(X,self._Ug) + ad.MatMul(S,self._Wg) + self._bg
        G=ad.Sigmoid(val_g)
        #print("G",G())
        val_r = ad.MatMul(X,self._Ur) + ad.MatMul(S,self._Wr) + self._br
        R=ad.Sigmoid(val_r)
        #print("R",R())
        val_h = ad.MatMul(X,self._Uh) + ad.MatMul(S*R,self._Wh) + self._bh

        H=ad.Sigmoid(val_h)
        #print("H",H())

        S_New = ((ad.Variable(np.ones_like(G.eval()))- G ) * H) + (Z*S)
        #print("Snew",S_New())
        
        #val = (-G * ((ad.Variable(np.ones_like(G.eval()))- G ) * H) * (self._Ug+self._Wg*self._Wg*temp)) + (((ad.Variable(np.ones_like(G.eval()))- G ) * H*(ad.Variable(np.ones_like(H.eval()))- H ))*(self._Uz+self._Wh*self._Wg*R*temp)+ (self._Wg*S*(ad.Variable(np.ones_like(R.eval()))- R ) * R*(self._Ug+self._Wg*self._Wg*temp))) +(Z*self._Wg*temp) + (S*(self._Ug+self._Wg*self._Wg*temp))
        #print(val())


        return S_New
    def set_params_layer(self,params):
        assert self._Uz.shape == params[0].shape and self._Ug.shape == params[1].shape and self._Ur.shape == params[2].shape and self._Uh.shape == params[3].shape \
            and self._Wz.shape == params[4].shape and self._Wg.shape == params[5].shape and self._Wr.shape == params[6].shape and self._Wh.shape == params[7].shape \
                and self._bz.shape == params[8].shape and self._bg.shape == params[9].shape and self._br.shape == params[10].shape and self._bh.shape == params[11].shape
        self._Uz.value = params[0]

        self._Ug.value = params[1]
        self._Ur.value = params[2]
        self._Uh.value = params[3]
        self._Wz.value = params[4]
        self._Wg.value = params[5]
        self._Wr.value = params[6]
        self._Wh.value = params[7]
        self._bz.value = params[8]
        self._bg.value = params[9]
        self._br.value = params[10]
        self._bh.value = params[11]
    def get_weights_layer(self):
        return [self._Uz,self._Ug,self._Ur,self._Uh,self._Wz,self._Wg,self._Wr,self._Wh,self._bz,self._bg,self._br,self._bh]
class NeuralNetLSTM():
    def __init__(self,number_of_units,number_of_layers,input_dim,output_dim):
        assert isinstance(number_of_units,int) and number_of_units>=2 and number_of_layers >=0 and isinstance(number_of_layers,int) and isinstance(input_dim,int) and input_dim>0 and isinstance(output_dim,int) and output_dim>0 

        self.number_of_units= number_of_units
        self.number_of_layers= number_of_layers
        self.input_dim = input_dim
        self.output_dim = output_dim
        #self.X=X
    
        self._W = ad.Variable(xavier(self.input_dim,self.number_of_units),name = "W")
        self._B = ad.Variable(xavier(1,self.number_of_units),name = "B")
        self._Wf = ad.Variable(xavier(self.number_of_units,self.output_dim),name = "Wf")
        self._Bf = ad.Variable(xavier(1,self.output_dim),name = "Bf")

        
        self.layer1 = lstm_layer(self.input_dim,self.number_of_units)
        self.layers = []
        self.layers.append(self.layer1)
        

        
        
        for i in range(self.number_of_layers):
            self.layers.append(lstm_layer(self.input_dim,self.number_of_units))
            
        
    
    def set_weights(self,params_of_layers):
        #self._W.value = params_of_layers[0]()
        self._B.value = params_of_layers[0]
        layer_params =[]
        iter = 1
        

        for i in range(self.number_of_layers + 1):
            self.layers[i].set_params_layer([param for param in params_of_layers[iter:iter+12]])
            iter = iter + 12

        self._Wf.value = params_of_layers[-2] 
        self._Bf.value = params_of_layers[-1]
    def get_weights(self):
        return_params = []
        #return_params.append(self._W)
        return_params.append(self._B)
        for i in range(self.number_of_layers + 1):
            return_params = return_params + (self.layers[i].get_weights_layer())
        return_params.append(self._Wf)
        return_params.append(self._Bf)
        return return_params
    def output(self,X):
        S = ad.Sigmoid(ad.MatMul(X,self._W) + self._B)
        
        
        #print("S:",S())
        S1 = self.layer1.output_layer(S,X)
        S_list = []
        S_list.append(S1)
        for i in range(self.number_of_layers):
            S_list.append(self.layers[i].output_layer(S_list[i],X))


        S_final = S_list[-1]
        #print(S_final.shape)
        #print(self.Wf.shape)
        #print(self.Bf.shape)
        val = ad.MatMul(S_final,self._Wf) + self._Bf
        #print("The output:",val())
        return val
        
def z(model,points):
    return model.output(points)
def loss_domain(model,points):
    """
    calculate loss at all the points. 
    three terms: 
    L1 : domain 
    L2: Initial Condition
    L3 : Boundary Condition 
    """
    t= ad.Variable(np.array([points[0]]),name = "t")
    x= ad.Variable(np.array([points[0]]),name = "x")

    points = ad.Reshape(ad.Concat(t,x,0),(1,2))
    u = model.output(points)
    du_dt,du_dx = ad.grad(u,[t,x])
    print("du_dt",du_dt())
    print("du_dx",du_dx())
    
    d2u_dx2 = diff_n_times(u,x,2)
    print("d2u_dx2",d2u_dx2())
    total_loss = du_dt + u*du_dx - (0.01/np.pi)*d2u_dx2
    print("loss",total_loss())

    return total_loss
if __name__ == "__main__":
    model = NeuralNetLSTM(10,1,1,1)
    print([i() for i in model.get_weights()])
    
    #print(model.output(np.array([[0,1]])))

    


    



"""
    model=NeuralNetLSTM(2,0,2,1)
    t = ad.Variable(np.array([0.75]),name = "t")
    x = ad.Variable(np.array([0.75]),name="x")
    point = ad.Reshape(ad.Concat(t,x,0),(1,2))
    print("output:",model.output(point)())
    gradwf = ad.grad(model.output(point),[x])[0]
    grad1 = ad.grad(model.output(point),[x])[0]
    grad2 = ad.grad(model.output(point),[t])[0]
    print("dX",grad1())
    print("dt",grad2())
    grad3 = ad.grad(model.output(point),[x,t])
    loss = loss_domain(model,[0.75,0.75])
    
    
   
    

    



    
    x=ad.Variable(np.array([1]),name ="x")
    
    t=ad.Variable(np.array([1]),name = "t")
    points1 = ad.Variable(np.array([1]),name="points1")
    #points=ad.Concat()
    points2 = ad.Variable(np.array([[1],[1]]),name="points2")
    W = ad.Variable(np.array([2]),name="W")
    #B = ad.Variable()
    q = ad.Concat(x,t,0)
    
    

    e = ad.Variable(np.full((2,2),1.5),name="e")
    q = ad.Reshape(q,(1,2))
    r= ad.MatMul(q,e)
    
    
    model=NeuralNetLSTM(2,0,2,1)
    point = ad.Variable(np.array([[0,0]]),name = "point")
    val = model.output(q)
    [dx,dt] = ad.grad(val,[x,t])
    

    d2x=diff_n_times(val,x,2)
    loss = loss_domain(model,[0.5,0.5])
    
    
    dx = ad.grad(loss,[x])

"""
    



    
    
    

    
    
    
    
    

    

    
    
    
    

    


        







    




        
    

        

        


    





