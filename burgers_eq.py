import autodiff as ad 
import numpy as np 
import matplotlib.pyplot as plt 
from NN_architecture import NeuralNetLSTM,lstm_layer
import matplotlib.pyplot as plt 
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
    n=50
    X_interior = np.random.uniform(low=X_low,high=X_high,size=50)
    t_interior = np.random.uniform(low=t_low,high=t_high,size=50)
    t_at_lower_boundary=np.random.uniform(low=t_low,high=t_high,size=15)
    t_at_upper_boundary = np.random.uniform(low=t_low,high=t_high,size=15)
    X_at_initial_condition= np.random.uniform(low=X_low,high=X_high,size=30)





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
    u = (points()[0][0])*(1+points()[0][1])*(1-points()[0][1])*z(model,points) - np.sin(np.pi*points()[0][0])
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
X_interior,t_interior,t_at_lower_boundary,t_at_upper_boundary,X_at_initial_condition = sampler(X_low,X_high,t_low,t_high)
samplings_domain = np.vstack((t_interior.transpose(),X_interior.transpose())).transpose()
samplings_initial = np.vstack((np.zeros_like(X_at_initial_condition.transpose()),X_at_initial_condition.transpose())).transpose()
samplings_boundary_lower = np.vstack((t_at_lower_boundary.transpose(),-1*np.ones_like(t_at_lower_boundary.transpose()))).transpose()
samplings_boundary_upper =np.vstack((t_at_upper_boundary.transpose(),np.ones_like(t_at_upper_boundary.transpose()))).transpose()


model = NeuralNetLSTM(20,2,2,1)
#model.set_weights(params)
L1= ad.Variable(0,name="L1")
L2= ad.Variable(0,name="L2")
L3= ad.Variable(0,name="L3")
#val = u(model,samplings_boundary_lower[10])
for i in range(50):
    L1 =L1 +(loss_domain(model,samplings_domain[i]))
#for i in range(30):
    #L2 =L2 +(loss_initial(model,samplings_initial[i]))
#for i in range(15):
    #L3 =L3 +(loss_boundary(model,samplings_boundary_lower[i])) + (loss_boundary(model,samplings_boundary_upper[i]))

total_loss = (L1/50) #+ (L2/30) 
    

#model = NeuralNetLSTM(20,2,2,1)
#total_loss = loss(model.get_weights(),20)
print("initial loss:",total_loss())


params = model.get_weights()
epochs = 50
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
    #new_loss = (loss_domain(model,samplings_domain[1]))+loss_boundary(model,samplings_boundary_lower[1]) + loss_boundary(model,samplings_boundary_upper[1])+loss_initial(model,samplings_initial[1])
    #new_loss = loss(model.get_weights(),20)
    L1= ad.Variable(0,name="L1")
    L2= ad.Variable(0,name="L2")
    L3= ad.Variable(0,name="L3")
#val = u(model,samplings_boundary_lower[10])
    for i in range(40):
        L1 =L1 +(loss_domain(model,samplings_domain[i]))
    #for i in range(30):
        #L2 =L2 +(loss_initial(model,samplings_initial[i]))
    #for i in range(15):
        #L3 =L3 +(loss_boundary(model,samplings_boundary_lower[i])) + (loss_boundary(model,samplings_boundary_upper[i]))

    new_loss = (L1/40) #+ (L2/30) 
    if np.abs(new_loss()-total_loss()) < 0.1:
        print("The loss is minimum, you can now test !")

        break
    elif new_loss() > total_loss() :
        print("The gradients are exploding, anyway plot with this set ")
        break
    else: 
        print("Gradient Descent step taken for iteration :",j)
        print("The updated loss is:",new_loss())
        total_loss = new_loss
        params = new_params
if j == epochs -1:
    print("OOps, couldn't converge")

x_list = np.random.uniform(-1,1,100)
t_list = np.full_like(x_list,0.25)
y_list = []
for i in x_list:
    X = ad.Variable(np.array([[0.25,i]]))
    val = 0.25*(i+1)*(1-i)*model.output(X)()[0][0] - np.sin(np.pi*i)
    y_list.append(val)

plt.scatter(x_list,y_list) 
plt.show()