import autodiff as ad 
import numpy as np 
import matplotlib.pyplot as plt 
from NN_architecture import NeuralNetLSTM,lstm_layer
import matplotlib.pyplot as plt 
import random
import sys
sys.setrecursionlimit(5000)
def z(model,points):
    return model.output(points)

def diff_n_times(graph, wrt, n):
    for i in range(n):
        graph = ad.grad(graph, [wrt])[0]
    return graph



def sampler(X_low,X_high,t_low,t_high,n):
    """
    Sample Points at problem area and boundary
    Strategy = Uniform 

    """
    
    X_interior = np.random.uniform(low=X_low,high=X_high,size=n)
    t_interior = np.random.uniform(low=t_low,high=t_high,size=n)
    return X_interior,t_interior
    


def loss_domain(model,points):
    """
    calculate loss at all the points. 
    three terms: 
    L1 : domain 
    L2: Initial Condition
    L3 : Boundary Condition 
    """
    t= ad.Variable(np.array([points[0]]),name = "t")
    x= ad.Variable(np.array([points[1]]),name = "x")
    points = ad.Reshape(ad.Concat(t,x,0),(1,2))

    u = t*(1+x)*(1-x)*z(model,points) - ad.Sine(np.pi*x)
    du_dt,du_dx = ad.grad(u,[t,x])
    
    
    d2u_dx2 = diff_n_times(u,x,2)
    total_loss = du_dt + u*du_dx - (0.01/np.pi)*d2u_dx2
    



    return total_loss
def loss_initial(model,points):
    penalizer = ad.Variable(np.sin(np.pi*points[1]),name="penalizer")
    points = ad.Variable(np.array([points]))
    u = (points()[0][0])*(1+points()[0][1])*(1-points()[0][1])*z(model,points) - np.sin(np.pi*points()[0][0])
    

    loss = u + penalizer
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
def samplings(n):
    X_interior,t_interior = sampler(X_low,X_high,t_low,t_high,n)
    samplings_domain = np.vstack((t_interior.transpose(),X_interior.transpose())).transpose()
    return samplings_domain


model = NeuralNetLSTM(30,2,2,1)
loss_list =[]
resampler_epochs = 5
for k in range(resampler_epochs):  
    print("Sampling done for the iteration:",k)
    epochs = 30
    tol = ad.Variable(0.001,name="tol")
    lr= 0.00146
    samplings_domain = samplings(50)
    

    for j in range(epochs):
        L1= ad.Variable(0,name="L1")
        
        

    
        for i in range(50):
            L1.value =L1.value +(loss_domain(model,samplings_domain[i])())
        #rand = random.randint(0,39)
        init_loss = ad.Pow(L1,2)/50
        print("initial_loss",init_loss())
        
        
        

        for i in range(50):
            L1 = loss_domain(model,samplings_domain[i])
            total_loss = ad.Pow(L1,2)
            #print("loss:",total_loss())
            params = model.get_weights()
            grad_params = [0 for _ in params]
            grad_params= ad.grad(total_loss,params)

        
    
            new_params= [0 for _ in params]
            #print("gradients taken!")
            for i in range(len(params)):
                new_params[i] = params[i] - lr* grad_params[i]
            model.set_weights(new_params)
            

        
    
        L1= ad.Variable(0,name="L1")
        

        for i in range(50):
            L1.value =L1.value +(loss_domain(model,samplings_domain[i])())
        #L1 = L1 + loss_domain(model,samplings_domain[rand])
        new_loss = ad.Pow(L1,2)/50
        loss_list.append(new_loss()[0][0])
        if new_loss()[0][0] < 1e-2 or new_loss()[0][0]> init_loss()[0][0] :
            print("The loss is minimum for this sample!",new_loss()[0][0])

            break
    
        else: 
            print("Gradient Descent step taken for iteration :",j)
            print("Now, loss",new_loss())
            

x_list = np.random.uniform(-1,1,100)
t_list = np.full_like(x_list,0.25)
y_list = []
iter_list = [i for i in range(len(loss_list))]
for i in x_list:
    X = ad.Variable(np.array([[0.5,i]]))
    val = 0.5*(i+1)*(1-i)*model.output(X)()[0][0] - np.sin(np.pi*i)
    y_list.append(val)
fig, ax = plt.subplots(2)
ax[0].scatter(x_list,y_list,label="The Ans")
ax[1].scatter(iter_list,loss_list,label="Loss over iterations") 
plt.show()
