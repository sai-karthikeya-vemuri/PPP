import pytest
import numpy as np 
from autodiff.core.ops import *
from autodiff.core.node import *
import autodiff as ad
from autodiff.core.grad import grad



def test_ReduceSumToShape1_as_scalar():
    X= np.ones((5,5))
    X = Variable(X,"X")
    val = ReduceSumToShape(X,())
    assert val() == 25 and isinstance(val,ReduceSumKeepDims)

def test_ReduceSumToShape2_as_column():
    X = np.ones((5,5))
    X = Variable(X,"X")
    val = ReduceSumToShape(X,(5,))
    print(val())
    flag = np.array_equal(val(),[5., 5., 5., 5., 5.])
    assert flag == True and isinstance(val,ReduceSumKeepDims)


def test_ReduceSumToShape3_as_row():
    X = np.ones((5,5))
    X = Variable(X,"X")
    val = ReduceSumToShape(X,(1,5))
    print(val())
    flag = np.array_equal(val(),np.full((1,5),5.))
    assert flag == True and isinstance(val,ReduceSumKeepDims)

def test_add1_2equal():
    x = np.array([[1,2],[4,5]])
    y = np.array([[1,2],[4,5]])
    X = Variable(x,"X")
    Y = Variable(y,"Y")
    Z = x+y

    Z1 = Add(X,Y)
    dzdx,dzdy = grad(Z1,[X,Y])
    assert np.array_equal(Z,Z1()) == True and isinstance(Z1,Add) and np.array_equal(dzdx(),np.ones_like(x)) and np.array_equal(dzdy(),np.ones_like(x))


def test_add2_matrix_vector():
    x = np.ones((2,))
    y = np.array([[1,2],[4,5]])
    Z = x+y
    X = Variable(x,"X")
    Y = Variable(y,"Y")

    
    Z1 = Add(X,Y)
    dzdx,dzdy = grad(Z1,[X,Y])
    print(dzdx())
    print(np.full_like(x,1.0))
    assert np.array_equal(Z,Z1()) == True and isinstance(Z1,Add) and np.array_equal(dzdx(),np.full_like(x,2)) and np.array_equal(dzdy(),np.full_like(y,1.0,dtype=type(dzdy())))



def test_add3_matrix_vector_fractions():
    x = (np.ones((1,2)))
    y = np.array([[1,2.0],[-4,5/1541]])
    Z = x+y
    X =Variable(x,"X")
    Y = Variable(y,"Y")
    Z1 = Add(X,Y)
    
    dzdx,dzdy = grad(Z1,[X,Y])
    print(dzdx())
    assert np.array_equal(Z,Z1()) == True and isinstance(Z1,Add) and np.array_equal(dzdx(),np.full_like(x,2)) and np.array_equal(dzdy(),np.ones_like(y) )



def test_add4_irrational():
    x = np.pi
    y = 10
    Z = x+y
    X = Variable(x,"X")
    Y = Variable(y,"Y")
    Z1 = Add(X,Y)
    dzdx,dzdy = grad(Z1,[X,Y])

    assert np.array_equal(Z,Z1()) == True and isinstance(Z1,Add) and dzdx()==dzdy()==1 
"""
def test_add5_complex_meant_to_fail():
    X = np.pi
    print("This is meant to fail to show that Complex numbers are not supported")
    Y = Variable(10j)
    Z = X+Y
    #Z1 = Add(X,Y)
    
    assert  isinstance(Y,TypeError)
"""
def test_Mul1_irrational():
    x = 1
    y = np.pi
    Z = x*y
    X = Variable(x,"X")
    Y = Variable(y,"Y")
    Z1 = Mul(X,Y)
    dzdx,dzdy = grad(Z1,[X,Y])
    assert isinstance(Z1,Mul) and Z1()==np.pi and dzdx()==y and dzdy()==x

def test_Mul2_matrix_scalar():
    x = np.array([[1,2],[2,2]])
    y = 3
    
    Z = x*y
    X = Variable(x,"X")
    Y= Variable(y,"Y")
    Z1 = Mul(X,Y)
    dzdx,dzdy = grad(Z1,[X,Y])
    print(dzdx())
    print(dzdy())

    assert isinstance(Z1,Mul) and np.array_equal(Z1(),Z)==True and np.array_equal(dzdx(),np.full_like(x,y)) and np.array_equal(dzdy(),7)


def test_Mul3_matrix_vector():
    x = np.array([1,0])
    y = np.array([[1,np.pi],[0,-np.exp(1)]])
    Z = x*y
    X = Variable(x,"X")
    Y = Variable(y,"Y")
    Z1 = Mul(X,Y)
    dzdx,dzdy = grad(Z1,[X,Y])
    print(dzdx())
    print(dzdy())
    assert isinstance(Z1,Mul) and np.array_equal(Z1(),Z)==True and np.array_equal(dzdy(),np.array([[1.,0.],[1.,0.]])) and np.array_equal(dzdx(),np.array([1,np.pi-np.exp(1)]))

def test_Negate_zero():
    X = Variable(0,"X")
    Y = Negate(X)
    dy = grad(Y,[X])[0]
    assert isinstance(Y,Negate) and Y()==0 and dy()==-1

def test_Negate_array():
    x = np.array([[np.pi,2.0],[-3,4.2222222]])
    X = Variable(x,"X")
    Y = Negate(X)
    dy = grad(Y,[X])[0]


    assert isinstance(Y,Negate) and np.array_equal(Y(),-1*x) == True and np.array_equal(dy(),np.full((2,2),-1.)) 

def test_recipr_scalar():
    x= 5
    X = Variable(x,"X")
    Y= Recipr(X)
    dy = grad(Y,[X])[0]



    assert isinstance(Y,Recipr) and np.abs(Y() - (1/(5+1e-12))) < 0.0000001 and np.abs(dy()  - np.array(-0.04)) < 0.000000001

def test_recipr_irrational_array():
    x = np.array([[np.pi,22/7],[np.exp(-1),np.exp(2.5)]])
    X = Variable(x,"X")
    Y = Recipr(X)
    
    dy = grad(Y,[X])[0]
    print(dy())
    print(-np.reciprocal((x+1e-12)*(x+1e-12)))
    assert isinstance(Y,Recipr) and np.array_equal(Y(),np.reciprocal(x+1e-12)) and np.all(np.less(np.abs(dy()+np.reciprocal((x+1e-12)*(x+1e-12))) , np.full_like(dy(),0.0000001)))

def test_einsum_onearg_identity():
    X = np.random.randn(2,3,5,7)
    Xv = Variable(X,"Xv")
    Z = Einsum("ijkl->ijkl",Xv)
    assert isinstance(Z,Einsum) and np.array_equal(Z(),X) == True

def test_einsum_onearg_sum_1axis():
    X = np.random.randn(2,3,5,7)
    Xv = Variable(X,"Xv")
    Z = Einsum("ijkl->ijk",Xv)
    assert isinstance(Z,Einsum) and np.array_equal(Z(),np.einsum("ijkl->ijk",X)) == True

def test_einsum_onearg_sum_2axis():
    X = np.random.randn(2,3,5,7)
    Xv = Variable(X,"Xv")
    Z = Einsum("ijkl->ij",Xv)
    assert isinstance(Z,Einsum) and np.array_equal(Z(),np.einsum("ijkl->ij",X)) == True


def test_einsum_onearg_sum_3axis():
    X = np.random.randn(2,3,5,7)
    Xv = Variable(X,"Xv")
    Z = Einsum("ijkl->i",Xv)
    assert isinstance(Z,Einsum) and np.array_equal(Z(),np.einsum("ijkl->i",X)) == True

def test_einsum_onearg_sum_allaxis():
    X = np.random.randn(2,3,5,7)
    Xv = Variable(X,"Xv")
    Z = Einsum("ijkl->",Xv)
    assert isinstance(Z,Einsum) and np.array_equal(Z(),np.einsum("ijkl->",X)) == True

def test_einsum_matmul():

    x = np.array([[3,4],[52,6]])
    y = np.array([[59,56],[64,44]])
    z = np.dot(x,y)
    X = Variable(x,"X")
    Y = Variable(y,"Y")
    Z = Einsum("ij,jk->ik",X,Y)
    dzdx , dzdy = grad(Z,[X,Y])
    assert isinstance(Z,Einsum) and np.array_equal(Z(),z) and np.array_equal(dzdx(),np.einsum("ik,jk->ij",np.ones_like(z),y)) \
        and np.array_equal(dzdy(),np.einsum("ij,jk->jk",x,np.ones_like(z)))

def test_einsum_matmul3():
    x = np.array([[3,4],[52,6]])
    y = np.array([[59,56],[4,44]])
    w = np.array([[151,49],[65,98]])
    z = np.dot(np.dot(x,y),w)
    X = Variable(x,"X")
    Y = Variable(y,"Y")
    W = Variable(w,"W")
    Z = Einsum("ij,jk,kl->il",X,Y,W)
    dzdx,dzdy,dzdw = grad(Z,[X,Y,W]) 
    print(dzdx())
    print(np.einsum("il,jk,kl->ij",np.ones_like(z),y,w))
    assert isinstance(Z,Einsum) and np.array_equal(Z(),z) and np.array_equal(dzdx(),np.einsum("il,jk,kl->ij",np.ones_like(z),y,w)) \
        and np.array_equal(dzdy(),np.einsum("ij,il,kl->jk",x,np.ones_like(z),w)) and np.array_equal(dzdw(),np.einsum("ij,jk,il->kl",x,y,np.ones_like(z)))


def test_einsum_different_indices():
    x = np.array([[3,4],[52,6]])
    y = np.array([[59,54],[44,84]])
    w = np.array([[11,29],[75,9]])
    z = np.einsum("ij,jk,kl->ijl",x,y,w)
    X = Variable(x,"X")
    Y = Variable(y,"Y")
    W = Variable(w,"W")
    Z = Einsum("ij,jk,kl->ijl",X,Y,W)
    dzdx,dzdy,dzdw = grad(Z,[X,Y,W])
    assert isinstance(Z,Einsum) and np.array_equal(Z(),z) and np.array_equal(dzdx(),np.einsum("ijl,jk,kl->ij",np.ones_like(z),y,w)) \
        and np.array_equal(dzdy(),np.einsum("ij,ijl,kl->jk",x,np.ones_like(z),w)) and np.array_equal(dzdw(),np.einsum("ij,jk,ijl->kl",x,y,np.ones_like(z)))

def test_einsum_4thorder():
    x = np.array([[35,24],[52,6]])
    y = np.array([[59,56],[44,44]])
    w = np.array([[11,45],[75,28]])
    z = np.einsum("ij,jk,kl->ijkl",x,y,w)
    X = Variable(x,"X")
    Y = Variable(y,"Y")
    W = Variable(w,"W")
    Z = Einsum("ij,jk,kl->ijkl",X,Y,W)
    dzdx,dzdy,dzdw = grad(Z,[X,Y,W])
    assert isinstance(Z,Einsum) and np.array_equal(Z(),z) and np.array_equal(dzdx(),np.einsum("ijkl,jk,kl->ij",np.ones_like(z),y,w)) \
        and np.array_equal(dzdy(),np.einsum("ij,ijkl,kl->jk",x,np.ones_like(z),w)) and np.array_equal(dzdw(),np.einsum("ij,jk,ijkl->kl",x,y,np.ones_like(z)))


def test_pow_scalar():
    x = 3
    y = 3**2
    X = Variable(x,"X")
    Y = Pow(X,2)
    dy = grad(Y,[X])[0]

    assert isinstance(Y,Pow) and Y()==9 and dy()==6

def test_pow_scalar_irrational():
    x = 3
    y = 3**-np.pi
    X = Variable(x,"X")
    Y = Pow(X,-np.pi)
    dy = grad(Y,[X])[0]


    assert isinstance(Y,Pow) and Y()==y and dy()==-np.pi*3**(-np.pi-1)
def test_pow_array_with_scalar():
    x = np.random.rand(3,3,3)
    y = x**2
    X = Variable(x,"X")
    Y = Pow(X,2)
    dy = grad(Y,[X])[0]


    assert isinstance(Y,Pow) and np.array_equal(Y(),y) and np.array_equal(dy(),2*x)

def test_pow_array_with_itself():
    x = np.random.rand(3,3,3)
    y = x**x
    
    X = Variable(x,"X")
    Y = Pow(X,X)
    #print(Y())
    dy = grad(Y,[X])[0]
    print(dy())
    print((x**x)*(np.log(x)+1))


    assert isinstance(Y,Pow) and np.array_equal(Y(),y) and np.array_equal(dy(),(x**x)*(np.log(x+1e-12)+1))

def test_pow_array_with_another():
    x = np.random.rand(4,4,4)
    z = np.random.rand(4,4,4)
    X = Variable(x,"X")
    Z = Variable(z,"Z")
    y = x**z
    print(y)
    Y = Pow(X,Z)
    print(Y())
    dydx ,dydz= grad(Y,[X,Z])



    assert isinstance(Y,Pow) and np.array_equal(Y(),y) and np.array_equal(dydx(),z*(x**(z-1))) and np.array_equal(dydz(),(x**z)*np.log(x+1e-12))


def test_log_scalar():
    x = 5 
    y = np.log(x+1e-12)
    X = Variable(x,"X")
    Y = Log(X)
    dy = grad(Y,[X])[0]
    assert isinstance(Y,Log) and Y()==y and dy()==1/(5+1e-12)

def test_log_0():
    x = 0 
    y = np.log(x+1e-12)
    X = Variable(x,"X")
    Y = Log(X)
    dy = grad(Y,[X])[0]
    assert isinstance(Y,Log) and Y()==y and dy()==1/1e-12
def test_log_array():
    x = np.array([[np.pi,np.exp(1)],[232,848]])
    y = np.log(x+1e-12)
    X = Variable(x,"X")
    Y = Log(X)
    dy=grad(Y,[X])[0]
    print(dy())
    print(1/x)

    assert isinstance(Y,Log) and np.array_equal(Y(),y) and np.array_equal(dy(),1/(x+1e-12))

def test_identity():
    x = np.array([[np.pi,np.exp(1)],[232.3864641,-84.448]])
    X = Variable(x,"X")
    y = Identity(X)
    dy=grad(y,[X])[0]
    print(dy())
    print(y())
    
    assert isinstance(y,Identity) and np.array_equal(x,y()) and np.array_equal(dy(),np.full_like(x,1.))

def test_exp_scalar():
    x = 5 
    y = np.exp(x)
    X = Variable(x,"X")
    Y = Exp(X)
    dy=grad(Y,[X])[0]
    assert isinstance(Y,Exp) and Y()==y and dy()==y
def test_exp_array():
    x = np.random.rand(2,2)
    y = np.exp(x)
    X = Variable(x,"X")
    Y = Exp(X)
    dy=grad(Y,[X])[0]
    assert isinstance(Y,Exp) and np.array_equal(Y(),y) and np.array_equal(dy(),y)


def test_sine_scalar():
    x = np.pi/2
    y = np.sin(x)
    
    X = Variable(x,"X")
    Y = ad.Sine(X)
    dy = grad(Y,[X])[0]
    print(dy())
    assert isinstance(Y,Sine) and y==Y() and dy()==np.cos(x)

def test_sine_array():
    x = np.random.rand(4,5,6)
    y = np.sin(x)
    X = Variable(x,"X")
    Y = ad.Sine(X)
    dy = grad(Y,[X])[0]
    print(dy())
    assert isinstance(Y,Sine) and np.array_equal(Y(),y) and np.array_equal(dy(),np.cos(x))

def test_cosine_array():
    x = np.random.rand(4,5)
    y = np.cos(x)
    X = Variable(x,"X")
    Y = ad.Cosine(X)
    
    dy = grad(Y,[X])[0]
    print(dy())
    print(-np.sin(x))
    assert isinstance(Y,Cosine) and np.array_equal(Y(),y) and np.array_equal(dy(),-np.sin(x))

def test_cosine_scalar():
    x = 0
    y = np.cos(x)
    Y = Cosine(x)
    dy = grad(Y,[x])[0]
    print(dy())
    print(-np.sin(x))
    assert isinstance(Y,Cosine) and y==Y() and dy()==-np.sin(x)

def test_tan_scalar():
    x = np.pi/2
    X = Variable(x,"X")
    y = np.tan(x)
    Y = Tan(X)
    dy = grad(Y,[X])[0]
    print(dy())
    temp=((1.0/(np.cos(x+1e-12)**2)))
    print(temp)
    assert isinstance(Y,Tan) and y == Y() and (temp-dy()) < 0.000000000001
def test_tan_scalar1():
    x = 0
    X = Variable(x,"X")
    y = np.tan(x)
    Y = Tan(X)
    dy = grad(Y,[X])[0]
    print(dy())
    temp=((1.0/(np.cos(x+1e-12)**2)))
    print(temp)
    assert isinstance(Y,Tan) and y == Y() and (temp-dy()) < 0.000000000001
def test_tan_array():
    x = np.random.rand(4,5)
    y = np.tan(x)
    X = Variable(x,"X")
    Y = Tan(X)    
    dy = grad(Y,[X])[0]
    print(dy())
    temp=((1.0/(np.cos(x)**2)))
    print(temp)
    assert isinstance(Y,Tan) and np.array_equal(Y(),y) and np.all(np.less(np.abs(dy()-temp),0.0000000001))

def test_cosec_array():
    x = np.random.rand(4,5)
    y = 1.0/np.sin(x+1e-12)
    X = Variable(x,"X")
    Y = Cosec(X)    
    dy = grad(Y,[X])[0]
    print(dy())
    temp=((1.0/(np.sin(x+1e-12)*np.tan(x+1e-12))))
    print(temp)
    assert isinstance(Y,Cosec) and np.array_equal(Y(),y) and np.all(np.less(np.abs(dy()+temp),0.0000000001))


def test_cosec_array_higher():
    x = np.random.rand(4,5,5)
    y = 1.0/np.sin(x+1e-12)
    X = Variable(x,"X")
    Y = Cosec(X)    
    dy = grad(Y,[X])[0]
    print(dy())
    temp=((1.0/(np.sin(x+1e-12)*np.tan(x+1e-12))))
    print(temp)
    assert isinstance(Y,Cosec) and np.array_equal(Y(),y) and np.all(np.less(np.abs(dy()+temp),0.0000000001))

def test_cosec_scalar():
    x = 0
    y = 1.0/(np.sin(x+1e-12))
    X = Variable(x,"X")
    Y = Cosec(X)    
    print(Y())
    print(y)
    dy = grad(Y,[X])[0]
    print(dy())
    temp=((1.0/(np.sin(-1e-12)*np.tan(x+1e-12))))
    print(temp)
    assert isinstance(Y,Cosec) and np.array_equal(Y(),y) and dy() < -1e23 and temp < -1e23


def test_cosec_scalar_1():
    x = np.pi/2
    y = 1.0/(np.sin(x+1e-12))
    X = Variable(x,"X")
    Y = Cosec(X)    
    print(Y())
    print(y)
    dy = grad(Y,[X])[0]
    print(dy())
    temp=((-1.0/(np.sin(x+1e-12)*np.tan(x+1e-12))))
    print(temp)
    assert isinstance(Y,Cosec) and np.array_equal(Y(),y) and np.all(np.less(np.abs(dy()-temp),0.0000000001))


def test_sec_scalar_1():
    x = 0
    y = 1.0/(np.cos(x+1e-12))
    X = Variable(x,"X")
    Y = Sec(X)    
    print(Y())
    print(y)
    dy = grad(Y,[X])[0]
    print(dy())
    temp=np.tan(x+1e-12)/np.cos(x+1e-12)
    print(temp)
    assert isinstance(Y,Sec) and np.array_equal(Y(),y) and np.all(np.less(np.abs(dy()-temp),0.0000000001))

def test_sec_scalar():
    x = np.pi/2
    y = 1.0/(np.cos(x+1e-12))
    X = Variable(x,"X")
    Y = Sec(X)    
    print(Y())
    print(y)
    dy = grad(Y,[X])[0]
    print(dy())
    temp= Sec(X)*Tan(X)
    print(temp())
    assert isinstance(Y,Sec) and np.array_equal(Y(),y) and np.all(np.less(np.abs(dy()-temp()),0.0000000001))


def test_sec_array():
    x = np.random.randn(10,20)
    y = 1.0/(np.cos(x+1e-12))
    X = Variable(x,"X")
    Y = Sec(X)    
    print(Y())
    print(y)
    dy = grad(Y,[X])[0]
    print(dy())
    temp= Sec(X)*Tan(X)
    print(temp())
    assert isinstance(Y,Sec) and np.array_equal(Y(),y) and np.all(np.less(np.abs(dy()-temp()),0.0000000001))



def test_cot_scalar():
    x = np.pi/2
    y = 1.0/(np.tan(x+1e-12))
    X = Variable(x,"X")
    Y = Cot(X)    
    print(Y())
    print(y)
    dy = grad(Y,[X])[0]
    print(dy())
    temp= -1/((np.sin(x+1e-12)**2))
    print(temp)
    assert isinstance(Y,Cot) and np.array_equal(Y(),y) and np.all(np.less(np.abs(dy()-temp),0.0000000001))


def test_cot_scalar():
    x = 0
    y = 1.0/(np.tan(x+1e-12))
    X = Variable(x,"X")
    Y = Cot(X)    
    print(Y())
    print(y)
    dy = grad(Y,[X])[0]
    print(dy())
    temp= -1/((np.sin(x+1e-12)**2))
    print(temp)
    assert isinstance(Y,Cot) and np.array_equal(Y(),y) and dy() < -1e23 and temp < -1e23



def test_cot_array():
    x = np.random.randn(22,88)
    y = 1.0/(np.tan(x+1e-12))
    X = Variable(x,"X")
    Y = Cot(X)    
    print(Y())
    print(y)
    dy = grad(Y,[X])[0]
    print(dy())
    temp= -1/((np.sin(x+1e-12)**2))
    print(temp)
    assert isinstance(Y,Cot) and np.array_equal(Y(),y) and np.all(np.less(np.abs(dy()-temp),0.0000001))


def test_sigmoid_scalar():
    x = 0
    y = 1/(1+np.exp(-x))
    X =Variable(x,"X")
    Y = Sigmoid(X)
    dy = grad(Y,[X])[0]
    temp = Sigmoid(X)*(1-Sigmoid(X))
    assert isinstance(Y,Sigmoid) and Y()==y and dy()==temp()

def test_sigmoid_array():
    x = np.random.rand(3,3,4)
    y = 1/(1+np.exp(-x))
    X =Variable(x,"X")
    Y = Sigmoid(X)
    dy = grad(Y,[X])[0]
    temp = Sigmoid(X)*(1-Sigmoid(X))
    assert isinstance(Y,Sigmoid) and np.array_equal(Y(),y) and np.array_equal(dy(),temp())

def test_arcsin_scalar():
    x = 1
    y = np.arcsin(x)
    X =Variable(x,"X")
    Y = ArcSin(X)
    dy = grad(Y,[X])[0]
    print(dy())
    x = x 
    temp = 1/(np.sqrt(1-(x*x))+1e-12)
    print(temp)
    assert isinstance(Y,ArcSin) and Y()==y and dy()-temp < 1e-20

def test_arcsin_array():
    x = np.random.rand(5,5)
    y = np.arcsin(x)
    X =Variable(x,"X")
    Y = ArcSin(X)
    dy = grad(Y,[X])[0]
    print(dy())
    temp = 1/np.sqrt(1-(x*x))
    assert isinstance(Y,ArcSin) and np.array_equal(Y(),y) and np.all(np.less(np.abs(dy()-temp),0.000000001))

def test_arccos_scalar():
    x = 1
    y = np.arccos(x)
    X =Variable(x,"X")
    Y = ArcCos(X)
    dy = grad(Y,[X])[0]
    print(dy())
    x = x 
    temp = -1/(np.sqrt(1-(x*x))+1e-12)
    print(temp)
    assert isinstance(Y,ArcCos) and Y()==y and np.abs(dy()-temp) < 1e-20
def test_arccos_array():
    x = np.random.rand(5,5)
    y = np.arccos(x)
    X =Variable(x,"X")
    Y = ArcCos(X)
    dy = grad(Y,[X])[0]
    print(dy())
    temp = -1/np.sqrt(1-(x*x))
    assert isinstance(Y,ArcCos) and np.array_equal(Y(),y) and np.all(np.less(np.abs(dy()-temp),0.000000001))

def test_arctan_scalar():
    x = 1e20
    y = np.arctan(x)
    X =Variable(x,"X")
    Y = ArcTan(X)
    dy = grad(Y,[X])[0]
    print(dy())
    x = x 
    temp = 1/(1+(x*x)+1e-12)
    print(temp)
    assert isinstance(Y,ArcTan) and Y()==y and np.abs(dy()-temp) < 1e-20
    

def test_arctan_array():
    x = np.random.randn(5,5,5)
    y = np.arctan(x)
    X =Variable(x,"X")
    Y = ArcTan(X)
    dy = grad(Y,[X])[0]
    print(dy())
    temp = 1/(1+(x*x)+1e-12)
    assert isinstance(Y,ArcTan) and np.array_equal(Y(),y) and np.all(np.less(np.abs(dy()-temp),0.000000000001))

def test_arccot_array():
    x = np.random.randn(5,5,5)
    y = np.arctan(1/(x+1e-12))
    X =Variable(x,"X")
    Y = ArcCot(X)
    dy = grad(Y,[X])[0]
    print(dy())
    temp = -1/(1+(x*x)+1e-12)
    assert isinstance(Y,ArcCot) and np.array_equal(Y(),y) and np.all(np.less(np.abs(dy()-temp),0.0000000000001))


def test_arccot_scalar():
    x = 0
    y = np.arctan(1/(x+1e-12))
    X =Variable(x,"X")
    Y = ArcCot(X)
    print(Y())
    dy = grad(Y,[X])[0]
    print(dy())
    temp = -1/(1+(x*x)+1e-12)
    assert isinstance(Y,ArcCot) and np.array_equal(Y(),y) and np.all(np.less(np.abs(dy()-temp),0.0000000000001))

def test_arccosec_scalar():
    x = 1
    y = np.arcsin(1/(x+1e-12))
    X =Variable(x,"X")
    Y = ArcCosec(X)
    #print(Y())
    dy = grad(Y,[X])[0]
    print(dy())
    temp = -1/(x*np.sqrt((x*x)-1)+1e-12)
    print(temp)
    assert isinstance(Y,ArcCosec) and np.array_equal(Y(),y) and np.all(np.less(np.abs(dy()-temp),0.0000000000001))
def test_arccosec_array():
    x = np.random.uniform(1.1,1e20,(5,5))
    y = np.arcsin(1/(x+1e-12))
    X =Variable(x,"X")
    Y = ArcCosec(X)
    #print(Y())
    dy = grad(Y,[X])[0]
    print(dy())
    temp = -1/(x*np.sqrt((x*x)-1)+1e-12)
    print(temp)
    assert isinstance(Y,ArcCosec) and np.array_equal(Y(),y) and np.all(np.less(np.abs(dy()-temp),0.0000000000001))

def test_arcsec_scalar():
    x = 1
    y = np.arccos(1/(x+1e-12))
    X =Variable(x,"X")
    Y = ArcSec(X)
    #print(Y())
    dy = grad(Y,[X])[0]
    print(dy())
    temp = 1/(np.abs(x)*np.sqrt((x*x)-1)+1e-12)
    print(temp)
    assert isinstance(Y,ArcSec) and np.array_equal(Y(),y) and np.all(np.less(np.abs(dy()-temp),0.0000000000001))

def test_arcsec_array():
    x = np.random.uniform(1.1,1e20,(3,3))
    y = np.arccos(1/(x+1e-12))
    X =Variable(x,"X")
    Y = ArcSec(X)
    #print(Y())
    dy = grad(Y,[X])[0]
    print(dy())
    temp = 1/(np.abs(x)*np.sqrt((x*x)-1)+1e-12)
    print(temp)
    assert isinstance(Y,ArcSec) and np.array_equal(Y(),y) and np.all(np.less(np.abs(dy()-temp),0.0000000000001))


def test_abs_scalar():
    x = 0
    y = np.abs(x)
    X = Variable(x,"X")
    Y = Absolute(X)
    dy = grad(Y,[X])[0]
    temp = 0
    print(np.abs(temp-dy()))
    assert isinstance(Y,Absolute) and np.array_equal(Y(),y) and np.abs(dy()-temp) < 1e-10  

def test_abs_array():
    x= np.random.uniform(-1e20,1e20,(5,5))
    y = np.abs(x)
    X = Variable(x,"X")
    Y = Absolute(X)
    dy = grad(Y,[X])[0]
    temp = x / np.abs(x)
    assert isinstance(Y,Absolute) and np.array_equal(Y(),y) and np.all(np.less(np.abs(dy()-temp),0.0000000000001))


