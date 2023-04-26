import numpy as np
matriz1 = np.array([[5,2],[2,1]])
np.linalg.eigvals(matriz1)
np.linalg.eig(matriz1)

matriz2 = np.array([[3,2],[2,1]])
np.linalg.eigvals(matriz2)


#Pratica da aula 1
import pandas as pd
from sklearn.model_selection import train_test_split

dados = pd.read_excel("data1.xlsx",header=None)
dados

x = dados.iloc[:,0] #atributos
y = dados.iloc[:,1]
N = len(dados)

x_treino,x_teste,y_treino,y_teste = train_test_split(x,y,train_size=0.8,random_state=0) 

Nt= len(x_treino)
I=2  #grau do polinomio
maxit = 10*Nt
k =0 #numero de iterações

w= np.ones(I+1)
n = 0.01 #learning rate



def MSE(w,xt,yt):
    return(np.square(phi(w,xt) - yt).mean())

def phi(w,xt):
    I= len(w)
    nt=len(xt)
    powers=np.zeros(nt)
    vect= np.zeros(I)
    j=0
    for val in xt:
        for i in range(I):
            vect[i]= np.power(val,i)
        powers[j] = np.dot(w,vect)
        j+=1
    return powers

def grad_MSE(w,xt,yt):
    I=len(w)
    nt= len(xt)
    grad = np.zeros(I)
    phis= phi(xt,w)
    powers=p(xt,I)
    for i in range(nt):
        grad[i] += (phis(i)-yt[i])*powers[:,i]
    return grad

def p(xt,I):
    I = len(w)
    nt= len(xt)
    power= np.zeros((I,nt))
    for val in xt:
        for i in range(I):
            power[i,:]=np.power(val,i)
    return power

grad_MSE(w,x_treino,y_treino)

while k <= maxit:
    grad = grad_MSE(w,x_treino,y_treino)
    #direção de procura -grad
    d = -grad
    w += -n * grad






