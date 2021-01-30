"""
Solving the Euler beam equation for a simple cantilever beam with load at the end.
EI y'' = M 
M = -p*X
x= 10m
E(Youngs Modulus) = 200000MPa
I(Monent of Inertia in transverse direction) = 0.000005 m^4
p(Point load) = 10KN
y = deflection of the beam 
strategy:
-> Hard boundary condition assignments
-> loss is calculated for all the points and reduces to its mean square value before taking gradient steps
"""



#Import required packages
import autodiff as ad 
import numpy as np
from NN_architecture import NeuralNetLSTM,xavier,diff_n_times
import matplotlib.pyplot as plt 
from optimizers import *
#styling of plots
plt.style.use('dark_background')



def loss_calculator(model,points):
    """
    Calculates the loss within the domain nd boundary of the differential equation
    inputs:
    model: The Neural Network model to be trained
    points: The points at which loss should be calculated(should lie within the domain)
    returns:Mean Squared loss from all the points in domain [0,10]
    """
    X = ad.Variable(points,"X")
    
    val = (10-X)*(10-X)*model.output(X)
    #Force (S.I Units)
    p = 10000
    #Flexural Rigidity - EI
    F = 0.000005*200000*1000000
    temp = p/F

    f = (diff_n_times(val,X,2)) + ((temp*X))
    print(f.shape)
    lossd = ad.ReduceSumToShape(ad.Pow(f,2),())/100
    Xb = ad.Variable(np.array([[10]]))
    fb1 = model.output(Xb)
    lossb1 =  ad.ReduceSumToShape(ad.Pow(fb1,2),())
    fb2 = ad.grad(model.output(Xb),[Xb])[0]
    lossb2 =  ad.ReduceSumToShape(ad.Pow(fb2,2),())

    return lossd 




def sampler(n):
    """
    samples of random data points(uniformly distributed)
    inputs:
    n : number of data points

    returns array of size n  
    
    """
    np.random.seed(0)
    return np.reshape(np.random.uniform(0,10,n),(n,1))


#Instantiating model and optimizer
model = NeuralNetLSTM(10,1,1,1)
model.set_weights([xavier(i().shape[0],i().shape[1]) for i in model.get_weights()])
optimizer= Adamax(len(model.get_weights()))
epochs = 2000
x=sampler(100)


#-------------------------------------------------------Training--------------------------------------------------
for i in range(epochs):
    loss = loss_calculator(model,x)
    print("loss",loss())
    params = model.get_weights()
    grad_params = ad.grad(loss,params)
    new_params = optimizer([i() for i in params], [i() for i in grad_params])
    model.set_weights(new_params)
    loss2= loss_calculator(model,x)
    print("loss now",loss2())
    #Exit condition
    if loss2()< 1e-2:
        break
#-----------------------------------Plotting--------------------------------------
np.random.seed(0)
x_list = np.random.uniform(low=0,high=10,size=100)
def y(x,F,P):
    return ((-P*x**3)/(6*F)) + ((P*100*x)/(2*F)) - ((P*1000/(3*F)))
y_plot = y(x_list,0.000005*200000*1000000,10000)  
print(y_plot.shape)
y_list =[]
for i in x_list:
    X=ad.Variable(np.array([[i]]),name="X")
    val =(10-X)*(10-X)*model.output(X) 
    y_list.append(val()[0][0])
plt.plot(np.linspace(0,10,10),np.zeros(10),label="Beam before deflection")
plt.scatter(x_list,y_plot,marker="+",label="Analytical")
plt.scatter(x_list,y_list,marker="x",label="Predicted")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Cantilever Beam with 10kN Load at end, E= 0.0005m^4,I =200000Mpa")
plt.legend()
plt.show()