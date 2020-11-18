import autodiff as ad 
import numpy as np 
import matplotlib.pyplot as plt 
from NN_architecture import NeuralNetLSTM,lstm_layer
def z(model,points):
    return model.output(points)

def diff_n_times(graph, wrt, n):
    for i in range(n):
        graph = ad.grad(graph, [wrt])[0]
    return graph

def optimizer(model,loss,params,lr=0.01,tol=0.001):
    """
    Optimize the parameters using Stochastic Gradient Descent
    Parameters to be optimized : W,B,Wf,Bf,{Uz,Wz,bz},{Ug,Wg,bg},{Ur,Wr,br}
    """
    #lr = ad.Variable(lr,name="lr")
    #grad_params = ad.grad(loss,params)
    epochs = 100
    old_params = np.array(params)
    for i in range(epochs):
        grad_params = ad.grad(loss,params)
        params = old_params - np.multiply(np.full_like(grad_params,lr),grad_params)
        if np.absolute(old_params-params) < np.full_like(params,tol):
            break
        old_params = params
        model.set_weights(params)
        loss = loss_domain(model,[0.5,0.5])+loss_boundary(model,[0.5,1])+loss_initial(model,[0,0.5])
        print("the loss:",loss)
        if epochs % 10 == 0:
            print("Don't be tensed everything will be alright")
        
    if i ==100:
        print("failed to converge,atleast you tried. Now the loss is", loss)
    else:
        model.set_weights(params)





    return params

    
def sampler(X_low,X_high,t_low,t_high):
    """
    Sample Points at problem area and boundary
    Strategy = Uniform 

    """
    X_interior = np.random.uniform(low=X_low,high=X_high,size=100)
    t_interior = np.random.uniform(low=t_low,high=t_high,size=100)
    t_at_lower_boundary=np.random.uniform(low=t_low,high=t_high,size=25)
    t_at_upper_boundary = np.random.uniform(low=t_low,high=t_high,size=25)
    X_at_initial_condition= np.random.uniform(low=X_low,high=X_high,size=50)



    return X_interior,t_interior,t_at_lower_boundary,t_at_upper_boundary,X_at_initial_condition
    


def loss_domain(model,points):
    """
    calculate loss at all the points. 
    three terms: 
    L1 : domain 
    L2: Initial Condition
    L3 : Boundary Condition 
    """
    points = ad.Variable(np.array([points]))
    u = z(model,points)
    du= ad.grad(u,[points])[0]
    #print(du())
    du_dt = du()[0][0]
    du_dx =du()[0][1]
    d2u_dx2 = diff_n_times(u,points,2)()[0][1]
    total_loss = du_dt + u*du_dx - (0.01/np.pi)*d2u_dx2
    #total_loss=0



    return ad.Pow(total_loss,2)
def loss_initial(model,points):
    penalizer = ad.Variable(np.sin(np.pi*points[1]),name="penalizer")
    points = ad.Variable(np.array([points]))
    u = z(model,points)
    

    loss = u + penalizer
    return ad.Pow(loss,2) 
def loss_boundary(model,points):
    points = ad.Variable(np.array([points]))
    u = z(model,points)

    loss = u 
    return ad.Pow(loss,2) 
    

X_low = -1
X_high = 1
t_low = 0
t_high =1
#model = NeuralNetLSTM(X,10,3,2,1)
X_interior,t_interior,t_at_lower_boundary,t_at_upper_boundary,X_at_initial_condition = sampler(X_low,X_high,t_low,t_high)
samplings_domain = np.vstack((t_interior.transpose(),X_interior.transpose())).transpose()
samplings_initial = np.vstack((np.zeros_like(X_at_initial_condition.transpose()),X_at_initial_condition.transpose())).transpose()
samplings_boundary_lower = np.vstack((t_at_lower_boundary.transpose(),-1*np.ones_like(t_at_lower_boundary.transpose()))).transpose()
samplings_boundary_upper =np.vstack((t_at_upper_boundary.transpose(),np.ones_like(t_at_upper_boundary.transpose()))).transpose()

model = NeuralNetLSTM(10,3,2,1)
L1= ad.Variable(0,name="L1")
L2= ad.Variable(0,name="L2")
L3= ad.Variable(0,name="L3")
#val = u(model,samplings_boundary_lower[10])
for i in range(100):
    L1 =L1 +(loss_domain(model,samplings_domain[i]))
for i in range(50):
    L2 =L2 +(loss_initial(model,samplings_initial[i]))
for i in range(25):
    L3 =L3 +(loss_boundary(model,samplings_boundary_lower[i])) + (loss_boundary(model,samplings_boundary_upper[i]))

total_loss = L1/100 + L2/50 + L3/50
optimized_params = optimizer(model,total_loss,model.get_weights())
