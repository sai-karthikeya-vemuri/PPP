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

def optimizer(model,loss,lr=0.01,tol=0.001):
    """
    Optimize the parameters using Stochastic Gradient Descent
    Parameters to be optimized : W,B,Wf,Bf,{Uz,Wz,bz},{Ug,Wg,bg},{Ur,Wr,br}
    

    
        #if np.absolute(loss()-new_loss()) > tol:
            #loss = new_loss
        print("the loss:",loss())
        if epochs % 10 == 0:
            print("Don't be tensed everything will be alright")
        else:
            print("The model is converged, the minimized loss is :",loss())
            break
        
    if i ==100:
        print("failed to converge,atleast you tried. Now the loss is", loss())
    else:
        model.set_weights(new_params)
        """





    return 0
def updater(new_params):
    model = NeuralNetLSTM(10,3,2,1)
    model.set_weights(new_params)
    print("The parameters are also set now")
    return model
    
def sampler(X_low,X_high,t_low,t_high):
    """
    Sample Points at problem area and boundary
    Strategy = Uniform 

    """
    n=20
    X_interior = np.random.uniform(low=X_low,high=X_high,size=n)
    t_interior = np.random.uniform(low=t_low,high=t_high,size=n)
    t_at_lower_boundary=np.random.uniform(low=t_low,high=t_high,size=n)
    t_at_upper_boundary = np.random.uniform(low=t_low,high=t_high,size=n)
    X_at_initial_condition= np.random.uniform(low=X_low,high=X_high,size=n)





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

model = NeuralNetLSTM(20,2,2,1)
L1= ad.Variable(0,name="L1")
L2= ad.Variable(0,name="L2")
L3= ad.Variable(0,name="L3")
#val = u(model,samplings_boundary_lower[10])
for i in range(20):
    L1 =L1 +(loss_domain(model,samplings_domain[i]))
for i in range(10):
    L2 =L2 +(loss_initial(model,samplings_initial[i]))
for i in range(10):
    L3 =L3 +(loss_boundary(model,samplings_boundary_lower[i])) + (loss_boundary(model,samplings_boundary_upper[i]))

total_loss = L1/20 + L2/10 + L3/20


print("initial loss:",total_loss())


params = model.get_weights()
epochs = 150
tol = ad.Variable(0.001,name="tol")
lr= 0.001
grad_params = []
for j in range(epochs):
    for i in params:
        temp = ad.grad(total_loss,[i])[0]
        
        grad_params.append(temp)

    new_params = []
    for i in range(len(params)):
        temp = params[i] - lr*grad_params[i]
        #print(temp().shape)
        new_params.append(temp)
    model.set_weights(new_params)
    new_loss = (loss_domain(model,samplings_domain[1]))+loss_boundary(model,samplings_boundary_lower[1]) + loss_boundary(model,samplings_boundary_upper[1])+loss_initial(model,samplings_initial[1])
    print("Gradient Descent step taken for iteration :",j)
    print("The updated loss is:",new_loss())
    total_loss = new_loss
    params = new_params
"""  
for i in range(epochs):
    
        
    new_params = []
        
    for i in range(len(model.get_weights())):
        new_params.append(params[i] - lr* grad_params[i])
        
    #new_model = updater(new_params)
    #new_model.set_weights(new_params)
    #model.set_weights(new_model.get_weights())
    #print(new_model.output(np.array([[1,2]])))
            
    
        
        
        
    print("there is no problem in updating")
        
    print("gradient step taken , that should be a relief")
    #model.set_weights(params)
    print("Parameters updated")
    
    new_loss=(loss_domain(model,samplings_domain[1])+loss_boundary(model,samplings_boundary_lower[1])+loss_initial(model,samplings_initial[1]))
    print("the new loss:")
    print(new_loss())
    print("something is not righht with the setter")
    print("loss is also calculated")


"""