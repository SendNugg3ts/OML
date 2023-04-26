import numpy as np
import matplotlib.pyplot as plt
import csv

RNG = np.random.default_rng()


def read_csv(filepath: str):
    with open(filepath) as file:
        file_reader = csv.reader(file, delimiter=',')
        X, y = [], []
        i = 0
        for row in file_reader:
            if i == 0:
                row[0] = row[0][3:]
                i += 1
            X.append(float(row[0].replace(",",".")))
            y.append(float(row[1].replace(",",".")))
    return np.array(X), np.array(y)


def new_randint(used_numbers, high):
    idx = RNG.integers(low=0, high=high)
    while used_numbers.get(idx) is not None:
        idx = RNG.integers(low=0, high=high)
    return idx


def random_selection(X, y):
    train_X, train_y = [], []
    test_X, test_y = [], []
    used_numbers = {}
    for _ in range(80):
        idx = new_randint(used_numbers, 100)
        used_numbers[idx] = idx
        train_X.append(X[idx])
        train_y.append(y[idx])
    for _ in range(20):
        idx = new_randint(used_numbers, 100)
        used_numbers[idx] = idx
        test_X.append(X[idx])
        test_y.append(y[idx])
    return np.array(train_X), np.array(train_y), np.array(test_X), np.array(test_y)


def get_powers(X, I):
    powers = np.zeros((I+1, len(X)))
    for i in range(len(X)):
        for j in range(I+1):
            powers[j, i] = np.power(X[i], j)
    return powers


def get_model_val(X, w):
    powers = get_powers(X, len(w)-1)
    model_y = np.dot(w.T, powers)
    return np.array(model_y)


def MSE(X, y, w):
    return (np.square(get_model_val(X, w) - y)).mean()


def MSE_grad(X, y, w):
    powers = get_powers(X, len(w)-1)
    aux = get_model_val(X, w) - y
    return (2 * np.sum(aux * powers, axis=1))/len(y)


def armijo(X, y, w, descent):
    n = RNG.uniform(high=1)
    sigma = 0.1

    rhs = sigma * n * MSE_grad(X, y, w).T * descent
    lhs = MSE(X, y, w)
    while MSE(X, y, w + n*descent) > (lhs + rhs).all():
        n /= 2
        rhs = sigma * n * MSE_grad(X, y, w).T * descent
    return n


def grad_batch(X, y, I):
    w = np.zeros(I+1)
    k = 0
    while k <= 10 * len(X):
        descent = -MSE_grad(X, y, w)
        learning_rate = armijo(X, y, w, descent)
        w += learning_rate*descent
        k += 1
    return np.dot(w.T, get_powers(X, I))


def grad_esto(X, y, I):
    w = np.zeros(I+1)
    k = 0
    used_idxs = {}
    while k <= 10 * len(X):
        idx = new_randint(used_idxs, len(X))
        if len(used_idxs) == len(X)-1:
            used_idxs.clear()
        used_idxs[idx] = idx

        descent = -MSE_grad(X[idx:idx+1], y[idx:idx+1], w)
        learning_rate = armijo(X[idx:idx+1], y[idx:idx+1], w, descent)
        w += learning_rate*descent
        k += 1
    return np.dot(w.T, get_powers(X, I))


def gen_idxs(lo, hi):
    seen = set()
    idx_1, idx_2 = RNG.integers(low=lo, high=hi), RNG.integers(low=lo, high=hi)

    while True:
        seen.add((idx_1, idx_2))
        yield (idx_1, idx_2)
        idx_1, idx_2 = RNG.integers(low=lo, high=hi), RNG.integers(low=lo, high=hi)
        while (idx_1, idx_2) in seen:
            idx_1, idx_2 = RNG.integers(low=lo, high=hi), RNG.integers(low=lo, high=hi)


def grad_mini_batch(X, y, I):
    w = np.zeros(I+1)
    k = 0
    idxs = gen_idxs(0, len(X))
    while k <= 10 * len(X):
        idx_1, idx_2 = next(idxs)
        while idx_1 == idx_2:
            idx_1, idx_2 = next(idxs)
        if idx_1 > idx_2:
            aux = idx_1
            idx_1 = idx_2
            idx_2 = aux

        descent = -MSE_grad(X[idx_1:idx_2], y[idx_1:idx_2], w)
        learning_rate = armijo(X[idx_1:idx_2], y[idx_1:idx_2], w, descent)
        w += learning_rate*descent
        k += 1
    return np.dot(w.T, get_powers(X, I))


if __name__ == '__main__':
    X, y = read_csv("data1.csv")
    train_X, train_y, test_X, test_y = random_selection(X, y)
    power = 3
    batch = grad_batch(train_X, train_y, power)
    esto = grad_esto(train_X, train_y, power)
    mini_batch = grad_mini_batch(train_X, train_y, power)

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(train_X, batch, 'go', X, y, 'b^')
    axs[0, 0].set_title('Gradient Batch')

    axs[0, 1].plot(train_X, esto, 'ro', X, y, 'b^')
    axs[0, 1].set_title('Gradient Estocastic')

    axs[1, 0].plot(train_X, mini_batch, 'ys', X, y, 'b^')
    axs[1, 0].set_title('Gradient Mini-Batch')
    plt.show()
