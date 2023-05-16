import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score


# Carregar o conjunto de dados Iris
iris = sns.load_dataset('iris')


# Criar um gráfico de dispersão com marcadores diferenciados para a espécie setosa e as outras espécies
setosa = iris[iris['species'] == 'setosa']
Outras_especies = iris[iris['species'] != 'setosa']
sns.scatterplot(data=setosa, x='petal_length', y='sepal_width', marker='o', label='setosa')
sns.scatterplot(data=Outras_especies, x='petal_length', y='sepal_width', marker='v', label='outras espécies')
plt.legend(labels=['setosa', 'outras espécies'])
plt.show()


#Fazer label enconding aos dados
X = iris.drop(['species'], axis=1).values
y = iris['species'].values

def label_encoding(y):
    y_encoded = np.zeros((y.shape[0]))
    for i in range(y.shape[0]):
        if y[i] == "setosa":                
            y_encoded[i] = 1
        else:                
            y_encoded[i] = -1
    return y_encoded

y = label_encoding(y)


# Separar os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# DEFINIR OS PARAMETROS
C = 1
tol = 0.001
kernel_type = 'l'
sigma = 0.7
max_passes = 3
max_iter= 200
#Definir a função de kernel
def kernel(x1, x2, kernel_type, sigma):
    if kernel_type == 'l': #linear
        return np.dot(x1, x2.T)
    elif kernel_type == 'g': #gaussian
        return np.exp(-np.linalg.norm(x1 - x2) ** 2 / (2 * sigma ** 2))
    elif kernel_type == 'p': #polinomial
        return (np.dot(x1, x2.T) + 1) ** 2


#ALGORITMO SMO
def smo(X, y, C, tol, kernel_type, sigma, max_iter):
    n_samples, n_features = X.shape
    m = X.shape[0]
    # compute kernel matrix
    K = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            K[i, j] = kernel(X[i], X[j], kernel_type, sigma)
    # initialize alpha vector
    alpha = np.zeros(m)
    bias = 0
    it = 0
    passes = 0
    
    while passes < max_passes and it < max_iter:
        it += 1
        changed_alphas = 0
        for i in range(n_samples):
            Ei = np.sum(alpha * y * kernel(X[i], X,kernel_type,sigma)) + bias - y[i]
            yEi = y[i] * Ei
            if (alpha[i] < C and yEi < -tol) or (alpha[i] > 0 and yEi > tol):
                j = np.random.choice(np.delete(np.arange(n_samples), i))
                Ej = np.sum(alpha * y * kernel(X[j], X,kernel_type,sigma)) + bias - y[j]
                ai = alpha[i]
                aj = alpha[j]
                if y[i] != y[j]:
                    L = max(0, aj - ai)
                    H = min(C, C + aj - ai)
                else:
                    L = max(0, ai + aj - C)
                    H = min(C, ai + aj)
                if L == H:
                    continue
                eta = 2 * kernel(X[i], X[j],kernel_type,sigma) - kernel(X[i], X[i],kernel_type,sigma) - kernel(X[j], X[j],kernel_type,sigma)
                if eta >= 0:
                    continue
                alpha[j] = aj - y[j] * (Ei - Ej) / eta
                alpha[j] = max(L, alpha[j])
                alpha[j] = min(H, alpha[j])
                if abs(alpha[j] - aj) < tol:
                    continue
                alpha[i] = ai + y[i] * y[j] * (aj - alpha[j])
                b1 = bias - Ei - y[i] * (alpha[i] - ai) * kernel(X[i], X[i],kernel_type,sigma) - y[j] * (alpha[j] - aj) * kernel(X[i], X[j],kernel_type,sigma)
                b2 = bias - Ej - y[i] * (alpha[i] - ai) * kernel(X[i], X[j],kernel_type,sigma) - y[j] * (alpha[j] - aj) * kernel(X[j], X[j],kernel_type,sigma)
                if 0 < alpha[i] < C:
                    bias = b1
                elif 0 < alpha[j] < C:
                    bias = b2
                else:
                    bias = (b1 + b2) / 2
                changed_alphas += 1
        if changed_alphas == 0:
            passes += 1
        else:
            passes = 0
    return alpha, bias



# Executar a função smo nos dados de treino
alpha, bias = smo(X_train, y_train, C, tol, kernel_type, sigma)
# Calcular os multiplicadores de Lagrange e o bias
print(f'Multiplicadores de Lagrange: {alpha}')
print(f'Bias: {bias}')

#ENCONTRAR OS VETORES DE SUPORTE
sv = alpha > 10e-3
idx = np.arange(len(alpha))[sv]
alpha_sv = alpha[sv]
X_sv = X_train[sv]
y_sv = y_train[sv]



# calcular bias
bias = 0
for i in range(len(alpha_sv)):
    bias = y_sv[i] - np.sum(alpha_sv * y_sv * kernel(X_sv, X_sv[i], kernel_type, sigma))
bias /= len(alpha_sv)

# SVM - Vetores de suporte:
print("SVM - Vetores de suporte:")
print("  n:    alpha_n         Xsv                   Y_n")
for j in range(len(idx)):
    print(f"{idx[j]:4d}    {alpha_sv[j]:.4f}      x=[{X_sv[j,0]:.4f}, {X_sv[j,1]:.4f}]     {y_sv[j]:4}")



#CLASSIFICADOR 
Xx = np.transpose(X_test)
m, n = Xx.shape
Yp = np.zeros(n)

for i in range(n):
    K = kernel(X_sv, Xx[:, i], kernel_type, sigma)
    Yp[i] = np.sign(np.sum(alpha_sv * y_sv * K) + bias)  # Yp - predicted class by the classifier

ErrDv = np.sum(np.abs(Yp - np.transpose(y_test)))  # out-sample error == error with validation data
print('\n out-sample error: %.4e ' % ErrDv)



# Gráfico
Yt=y_train
Xt=X_train
Yv=y_test
Xv=X_test
In = np.where(Yt == -1)[0]
Ip = np.where(Yt == 1)[0]
plt.plot(Xt[In, 0], Xt[In, 1], 'r.', markersize=15, label='-1: Conjunto de treinamento')
plt.plot(Xt[Ip, 0], Xt[Ip, 1], 'b.', markersize=15, label='+1: Conjunto de treinamento')

# Gráfico dos support vectors
plt.plot(X[0, idx], X[1, idx], 'ko', markersize=8, label='Support Vectors')

# Opcional: Gráfico do conjunto de validação
flag = True  # flag=True para incluir o conjunto de validação no gráfico, flag=False para não incluir
if flag:
    In = np.where(Yv == -1)[0]
    Ip = np.where(Yv == 1)[0]
    plt.plot(Xv[In, 0], Xv[In, 1], 'ro', markersize=4, label='-1: Conjunto de validação')
    plt.plot(Xv[Ip, 0], Xv[Ip, 1], 'bo', markersize=4, label='+1: Conjunto de validação')

# Gráfico da fronteira de decisão
d = 0.1
x1_min, x1_max = Xt[:, 0].min() - 1, Xt[:, 0].max() + 1
x2_min, x2_max = Xt[:, 1].min() - 1, Xt[:, 1].max() + 1
#num_points = 50  # ajuste este valor como desejado
#x1Grid, x2Grid = np.meshgrid(np.linspace(x1_min, x1_max, num_points), np.linspace(x2_min, x2_max, num_points))

x1Grid, x2Grid = np.meshgrid(np.arange(min(Xt[:, 0]), max(Xt[:, 0]), d),
                             np.arange(min(Xt[:, 1]), max(Xt[:, 1]), d))


xGrid = np.column_stack([x1Grid.ravel(), x2Grid.ravel()])

Yp = np.zeros_like(x1Grid)
for i in range(xGrid.shape[0]):
    K = kernel(X_sv, np.expand_dims(xGrid[i], axis=0), kernel_type, sigma)
    Yp[i] = np.sum(alpha * y_sv * K)

Yp = Yp.reshape(x1Grid.shape)

contour = plt.contour(x1Grid, x2Grid, Yp, levels=[0], colors='k', linewidths=1)

# Configurações adicionais do gráfico
plt.title('SVM dual com SMO + fronteira de decisão')
plt.xlabel('Característica 1')
plt.ylabel('Característica 2')
plt.legend()
plt.axis('equal')
plt.show()



























#Yp = np.zeros_like(x1Grid)
#for i in range(xGrid.shape[0]):
    #K = kernel(X_sv, xGrid, kernel_type, sigma)
    #Yp = np.dot(K.T, alpha * y_sv) + bias

    #Yp[i] = np.sum(alpha * y_sv * K) + bias