#Versão em python do SMO
import numpy as np
import matplotlib.pyplot as plt


def kernel(X1, X2):
    global gm, kfunction
    
    if kfunction == 'l':    
        return np.dot(X1.T, X2)
    elif kfunction == 'g':  
        return np.exp(-gm * np.sum((X1[:, None] - X2[:, :])**2, axis=2))
    elif kfunction == 'p':  
        return (1 + np.dot(X1.T, X2))**2
    else:
        raise ValueError('Invalid kernel function.')


# Read the data.
# Data = np.loadtxt('ex1data1.csv', delimiter=',')    # 48 points; use linear kernel.
# Data = np.loadtxt('ex1data2.csv', delimiter=',')    # 48 points; use linear kernel.
# Data = np.loadtxt('ex2data1.csv', delimiter=',')    # 200 points; use kernel.
Data = np.loadtxt('ex2data2.csv', delimiter=',')    # 400 points; use kernel.

N, M = Data.shape
Xdata = Data[:, :M-1]
Ydata = Data[:, -1]
I = M - 1  


In = np.where(Ydata == -1)[0]
Ip = np.where(Ydata == 1)[0]
plt.plot(Xdata[In, 0], Xdata[In, 1], 'r.', markersize=15)
plt.plot(Xdata[Ip, 0], Xdata[Ip, 1], 'b.', markersize=15)
plt.legend(['-1: data set', '+1: data set'])
plt.title('Data set')
plt.axis('equal')
plt.show()


op = 2   
Nt = int(round(0.8 * N))
if op == 1:
    p = np.arange(N)
else:   
    p = np.random.permutation(N)

Xt = Xdata[p[:Nt], :]
Yt = Ydata[p[:Nt]]

Xv = Xdata[p[Nt:], :]
Yv = Ydata[p[Nt:]]

X = Xt.T  
y = Yt[:, None].T    

m, N = X.shape    
alpha = np.zeros((1, N))    
bias = 0    
it = 0   

C = 10    
tol = 1e-4    
kfunction = 'g'    
sigma = 1    
gm = 1 / (2 * sigma**2)    
maxit = 10    
max_passes = 1    
passes = 0

while (passes < max_passes and it < maxit):  
    it += 1
    changed_alphas = 0  # number of changed alphas
    # N = len(y)                     # number of support vectors
    for i in range(N):  # for each alpha_i
        Ei = sum(alpha * y * K(X, X[:, i])) + bias - y[i]
        yE = Ei * y[i]
        if (alpha[i] < C and yE < -tol) or (alpha[i] > 0 and yE > tol):  # KKT violation
            for j in [k for k in range(N) if k != i]:  # for each alpha_j not equal alpha_i
                Ej = sum(alpha * y * K(X, X[:, j])) + bias - y[j]
                ai = alpha[i]  # alpha_i old
                aj = alpha[j]  # alpha_j old
                if y[i] == y[j]:  # s=y_i y_j=1
                    L = max(0, alpha[i] + alpha[j] - C)
                    H = min(C, alpha[i] + alpha[j])
                else:  # s=y_i y_j=-1
                    L = max(0, alpha[j] - alpha[i])
                    H = min(C, C + alpha[j] - alpha[i])
                if L == H:  # skip when L==H
                    continue
                eta = 2 * K(X[:, j], X[:, i]) - K(X[:, i], X[:, i]) - K(X[:, j], X[:, j])
                alpha[j] = alpha[j] + y[j] * (Ej - Ei) / eta  # update alpha_j
                if alpha[j] > H:
                    alpha[j] = H
                elif alpha[j] < L:
                    alpha[j] = L
                if np.linalg.norm(alpha[j] - aj) < tol:  # skip if no change
                    continue
                alpha[i] = alpha[i] - y[i] * y[j] * (alpha[j] - aj)  # find alpha_i
                bi = bias - Ei - y[i] * (alpha[i] - ai) * K(X[:, i], X[:, i]) \
                     - y[j] * (alpha[j] - aj) * K(X[:, j], X[:, i])
                bj = bias - Ej - y[i] * (alpha[i] - ai) * K(X[:, i], X[:, j]) \
                     - y[j] * (alpha[j] - aj) * K(X[:, j], X[:, j])
                if 0 < alpha[i] and alpha[i] < C:
                    bias = bi
                elif 0 < alpha[j] and alpha[j] < C:
                    bias = bj
                else:
                    bias = (bi + bj) / 2
                changed_alphas += 1  # one more alpha changed
    if changed_alphas == 0:  # no more changed alpha, quit
        # break
        passes += 1
    else:
        passes = 0