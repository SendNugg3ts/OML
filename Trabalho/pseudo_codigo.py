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



#Definir a função de kernel
def kernel(x1, x2, kernel_type, sigma):
    if kernel_type == 'l': #linear
        return np.dot(x1, x2.T)
    elif kernel_type == 'g': #gaussian
        return np.exp(-np.linalg.norm(x1 - x2) ** 2 / (2 * sigma ** 2))
    elif kernel_type == 'p': #polinomial
        return (np.dot(x1, x2.T) + 1) ** 2



def smo(X, y, C, tol, kernel_type, sigma, max_iter=100):
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
    
    while passes < 1 and it < max_iter:
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



#definir os parametros
C = 1
tol = 0.001
kernel_type = 'g'
sigma = 0.3

# Executar a função smo nos dados de treino
alpha, bias = smo(X_train, y_train, C, tol, kernel_type, sigma)

# Função de decisão SVM
def predict(X_test, X_train, y_train, alpha, bias, kernel_type, sigma):
    y_pred = np.zeros(X_test.shape[0])
    for i, x in enumerate(X_test):
        s = 0
        for a, y, x_train in zip(alpha, y_train, X_train):
            s += a * y * kernel(x, x_train, kernel_type, sigma)
        y_pred[i] = np.sign(s + bias)
    return y_pred

# Prever as classes do conjunto de teste
y_pred = predict(X_test, X_train, y_train, alpha, bias, kernel_type, sigma)

# Avaliar o desempenho do modelo SVM
print(confusion_matrix(y_test, y_pred))
print('Accuracy:', accuracy_score(y_test, y_pred))


sv = alpha > 0
idx = np.arange(len(alpha))[sv]
alpha_sv = alpha[sv]
X_sv = X_train[sv]
y_sv = y_train[sv]


X_train_encoded = label_encoding(X_train)
X_test_encoded = label_encoding(X_test)

# calcular bias
bias = 0
for i in range(len(alpha_sv)):
    bias += y_sv[i] - np.sum(alpha_sv * y_sv * kernel(X_sv, X_sv[i], kernel_type, sigma))
bias /= len(alpha_sv)

# predição para todo o conjunto de teste
y_pred = predict(X_test_encoded, X_sv, y_sv, alpha_sv, bias, kernel_type, sigma)

y_test_encoded = np.where(y_test == -1, -1, 1)
cm = confusion_matrix(y_test_encoded, y_pred)
acc = accuracy_score(y_test_encoded, y_pred)

print(f"Acurácia: {acc:.2f}")
print(f"Matriz de confusão:\n{cm}")
#erro de validação
ErrDv = np.sum(np.abs(y_pred - y_test_encoded))
print(f'out-sample error: {ErrDv:.4e}')

# Obter os índices dos pontos de suporte vetor
sv = alpha > 0
idx_sv = np.arange(len(alpha))[sv]




##GRÁFICO
# Definir grade de pontos para a visualização da fronteira de decisão
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
Z = predict(np.c_[xx.ravel(), yy.ravel()], X_train[:, :2], y_train, alpha, bias, kernel_type, sigma)

Z = Z.reshape(xx.shape)

# Gráfico da fronteira de decisão e os pontos de treino

plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Paired, edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()


















