import csv
import numpy as np

dados= []
with open('exemplo1_D.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        dados.append(row)

def perceptron_learning_algorithm(dados):
    w = [0,0,0]
    t=0
    N= len(dados)
    while True:
        for n in dados:
            E = (1/N)*(np.sum((1/2)*abs(n[1]-[2])))
            
            if E == 0:
                w_star = w
                return w_star
            else:
                y_n = n[2]
                y_hat_n = np.sign(np.dot(w, dados[n,:-1]))
                
                if y_n != y_hat_n:
                    w = w + y_n * dados[n,:-1]
                
                t += 1
