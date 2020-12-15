import autodiff as ad 
import numpy as np 
import matplotlib.pyplot as plt 
from NN_architecture import NeuralNetLSTM,lstm_layer
import matplotlib.pyplot as plt 
import random
import sys
sys.setrecursionlimit(5000)


def diff_n_times(graph, wrt, n):
    for i in range(n):
        graph = ad.grad(graph, [wrt])[0]
    return graph



def sampler_domain(X_low,X_high,t_low,t_high,n):
    """
    Sample Points at problem area and boundary
    Strategy = Uniform 

    """
    #np.random.seed(0)
    X_interior = np.linspace(-1,1,n)
    t_interior = np.full_like(X_interior,0.5)
    return X_interior,t_interior
    
def sampler_initial(X_low,X_high,n):
    return np.linspace(-1,1,n)

def loss_domain(model,points):
    """
    calculate loss at all the points. 
    three terms: 
    L1 : domain 
    L2: Initial Condition
    L3 : Boundary Condition 
    """
    t= ad.Variable(np.array([0.5]),name = "t")
    x= ad.Variable(np.array([points[1]]),name = "x")
    points = ad.Reshape(ad.Concat(t,x,0),(1,2))

    
    du_dt,du_dx = ad.grad((1+x)*(1-x)*model.output(points),[t,x])
    
    
    d2u_dx2 = diff_n_times((1+x)*(1-x)*model.output(points),x,2)
    total_loss = (du_dt) + ((1+x)*(1-x)*model.output(points)*du_dx) - (0.00318309886*d2u_dx2)
    



    return ad.Pow(total_loss,2)
def loss_initial(model,points):
    t= ad.Variable(np.array([0]),name = "t")
    x= ad.Variable(np.array([points]),name = "x")
    points = ad.Reshape(ad.Concat(t,x,0),(1,2))

    
    loss = ((1+x)*(1-x)*model.output(points) + ad.Sine(np.pi * x))

    
    return ad.Pow(loss,2)
def loss_boundary(model,points):
    points = ad.Variable(np.array([points]))
    u = (1+points()[0][1])*(1-points()[0][1])*z(model,points)

     
    return ad.Pow(u,2) 


X_low = -1
X_high = 1
t_low = 0
t_high =1
#model = NeuralNetLSTM(X,10,3,2,1)
def Samplings_domain(n):
    X_interior,t_interior = sampler_domain(X_low,X_high,t_low,t_high,n)
    samplings_domain = np.vstack((t_interior.transpose(),X_interior.transpose())).transpose()
    return samplings_domain

def Samplings_initial(n):
    X_initial = sampler_initial(X_low,X_high,n)
    #samplings_initial = np.vstack((np.zeros_like(X_initial).transpose(),X_initial.transpose())).transpose()
    return X_initial
model = NeuralNetLSTM(50,5,2,1)
loss_list =[]
resampler_epochs = 10







for k in range(resampler_epochs):

    print("Sampling done for the iteration:",k)
    epochs = 50
    tol = ad.Variable(0.001,name="tol")
    lr= 0.00146
    samplings_domain = Samplings_domain(100)
    samplings_initial= Samplings_initial(100)
    #samplings = samplings_initial(50)
    optimizer = ad.Adam(len(model.get_weights()))
    for j in range(epochs):
        L1= ad.Variable(0,name="L1")
        #L2=ad.Variable(0,"L2")
        for i in range(100):
            L1.value =L1.value +(loss_domain(model,samplings_domain[i])()) +  (loss_initial(model,samplings_initial[i])())
            #L2.value =L2.value +(loss_initial(model,samplings[i])())
        
        init_loss = L1/100
        print("initial_loss",init_loss())
        for i in range(1):
            
            #L2 = loss_initial(model,samplings[i])
            #total_loss = L1
            params = model.get_weights()
            grad_params = [0 for _ in params]
            grad_params= ad.grad(loss_domain(model,samplings_domain[i]) + loss_initial(model,samplings_initial[i]),params)

        
    
            new_params= [0 for _ in params]
            #print("gradients taken!")
            """
            for i in range(len(params)):
                new_params[i] = params[i] - lr* grad_params[i]
            """
            new_params = optimizer([i() for i in params], [i() for i in grad_params])
            model.set_weights(new_params)
    
        L3= ad.Variable(0,name="L3")
        #L4= ad.Variable(0,name="L4")
        for i in range(100):
            L3.value =L3.value +(loss_domain(model,samplings_domain[i])()) + (loss_initial(model,samplings_initial[i])())
            #L4.value =L4.value +(loss_initial(model,samplings[i])())
    
        new_loss = L3/100
        loss_list.append(new_loss()[0][0])
        if new_loss()[0][0] < 1e-4 or new_loss()[0][0]> init_loss()[0][0] or np.abs(new_loss()[0][0]-init_loss()[0][0]) < 0.00001:
            print("The loss is minimum for this sample!",new_loss()[0][0])
            break    
        else: 
            print("Gradient Descent step taken for iteration :",j)
            print("Now, loss",new_loss())
            
x_list = np.linspace(-1,1,100)
y1_list = []
y2_list =[]
y3_list = []
iter_list = [i for i in range(len(loss_list))]
for i in x_list:
    X = ad.Variable(np.array([[0.5,i]]))
    val1 = (i+1)*(1-i)*(model.output(X)()[0][0])
    
    y1_list.append(val1)
    
plt.scatter(x_list,y1_list)
plt.legend()
plt.show()
