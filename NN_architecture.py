"""
I am going to solve the burgers equation here using the deep NN
and at present thr autodiff package by Bruno Gavranovic'

"""
import numpy as np 
import autodiff as ad 

def diff_n_times(graph, wrt, n):
    for i in range(n):
        graph = ad.grad(graph, [wrt])[0]
    return graph
def xavier(input_dim,output_dim):
    r = 4*np.sqrt(6/(input_dim+output_dim))
    
    def random_generator(low,high,input_dim,output_dim):
        return np.random.uniform(low=low,high=high,size=(input_dim,output_dim))
    value = random_generator(-r,+r,input_dim,output_dim)


    return np.array(value)
class lstm_layer():

    def __init__(self,input_dim,output_dim):
        #self.inputs = inputs
        self.input_dim = input_dim
        self.output_dim= output_dim
        self.Uz = ad.Variable(xavier(self.input_dim,self.output_dim),name="Uz")
        self.Ug = ad.Variable(xavier(self.input_dim,self.output_dim),name="Ug")
        self.Ur = ad.Variable(xavier(self.input_dim,self.output_dim),name="Ur")
        self.Uh = ad.Variable(xavier(self.input_dim,self.output_dim),name="Uh")
        self.Wz = ad.Variable(xavier(self.output_dim,self.output_dim),name="Wz")
        self.Wg = ad.Variable(xavier(self.output_dim,self.output_dim),name="Wg")
        self.Wr = ad.Variable(xavier(self.output_dim,self.output_dim),name="Wr")
        self.Wh = ad.Variable(xavier(self.output_dim,self.output_dim),name="Wh")
        self.bz = ad.Variable(xavier(1,self.output_dim),name = "bz")
        self.bg = ad.Variable(xavier(1,self.output_dim),name = "bg")
        self.br = ad.Variable(xavier(1,self.output_dim),name = "br")
        self.bh = ad.Variable(xavier(1,self.output_dim),name = "bh")
    def output_layer(self,S_Old,X):
        S=S_Old
        val_z = ad.MatMul(X,self.Uz) + ad.MatMul(S,self.Wz) + self.bz
        Z=ad.Sigmoid(val_z)
        val_g = ad.MatMul(X,self.Ug) + ad.MatMul(S,self.Wg) + self.bg
        G=ad.Sigmoid(val_g)
        val_r = ad.MatMul(X,self.Ur) + ad.MatMul(S,self.Wr) + self.br
        R=ad.Sigmoid(val_r)
        val_h = ad.MatMul(X,self.Uh) + ad.MatMul(S*R,self.Wh) + self.bh

        H=ad.Sigmoid(val_h)

        S_New = ((ad.Variable(np.ones_like(G.eval()))- G ) * H) + (Z*S)


        return S_New
    def set_params_layer(self,Uz,Ug,Ur,Uh,Wz,Wg,Wr,Wh,bz,bg,br,bh):
        self.Uz = Uz 
        self.Ug = Ug
        self.Ur = Ur
        self.Uh = Uh 
        self.Wz = Wz
        self.Wg = Wg 
        self.Wr = Wr 
        self.Wh = Wh 
        self.bz = bz 
        self.bg = bg 
        self.br = br 
        self.bh = bh
    def get_weights_layer(self):
        return self.Uz,self.Ug,self.Ur,self.Uh,self.Wz,self.Wg,self.Wr,self.Wh,self.bz,self.bg,self.br,self.bh
class NeuralNetLSTM():
    def __init__(self,number_of_units,number_of_layers,input_dim,output_dim):
        self.number_of_units= number_of_units
        self.number_of_layers= number_of_layers
        self.input_dim = input_dim
        self.output_dim = output_dim
        #self.X=X
    
        self.W = ad.Variable(xavier(self.input_dim,self.number_of_units),name = "W")
        self.B = ad.Variable(xavier(1,self.number_of_units),name = "B")
        self.Wf = ad.Variable(xavier(self.number_of_units,1),name = "Wf")
        self.Bf = ad.Variable(xavier(1,self.output_dim),name = "Bf")

        
        self.layer1 = lstm_layer(self.input_dim,self.number_of_units)
        self.layers = []
        self.layers.append(self.layer1)
        

        
        
        for i in range(self.number_of_layers):
            self.layers.append(lstm_layer(self.input_dim,self.number_of_units))
            
        
    def output(self,X):
        S = ad.Sigmoid(ad.MatMul(X,self.W) + self.B)
        S1 = self.layer1.output_layer(S,X)
        S_list = []
        S_list.append(S1)
        for i in range(self.number_of_layers):
            S_list.append(self.layers[i].output_layer(S_list[i],X))


        S_final = S_list[-1]
        #print(S_final.shape)
        #print(self.Wf.shape)
        #print(self.Bf.shape)
        val = ad.MatMul(S_final,self.Wf) + self.Bf
        
        return val
    def set_weights(self,params_of_layers):
        self.W = params_of_layers[0]
        self.B = params_of_layers[1]
        layer_params =[]
        iter = 2
        

        for i in range(self.number_of_layers + 1):
            self.layers[i].set_params_layer([param for param in params_of_layers[iter:iter+12]])
            iter = iter + 12

        self.Wf = params_of_layers[-2] 
        self.Bf = params_of_layers[-1]
    def get_weights(self):
        return_params = []
        return_params.append(self.W)
        return_params.append(self.B)
        for i in range(self.number_of_layers + 1):
            return_params.append(self.layers[i].get_weights_layer())
        return_params.append(self.Wf)
        return_params.append(self.Bf)
        return return_params

if __name__ == "__main__":

    
    
    X=ad.Variable(np.array([[1,0]]),name="X")
    model=NeuralNetLSTM(10,3,2,1)
    val = model.output(X)
    gradx = ad.grad(val,[X])[0]
    print(gradx()[0][0])
    
    

    


        







    




        
    

        

        


    





