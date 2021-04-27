import numpy as np
import pandas as pd


df = pd.read_csv('data.csv')

X = df.iloc[0:100, [0,2]].values
y = df.iloc[0:100, 4].values
Y = np.where(y == 'Iris-setosa', 1, 0)

shuffle = np.arange(len(X))
np.random.shuffle(shuffle)
X = X[shuffle]
Y = Y[shuffle]


in_size = X.shape[1]
hidden_size = 80
out_size = len(Y.shape)

Win = np.zeros((1+in_size,hidden_size)) 
Win[0,:] = (np.random.randint(-2, 2, size = (hidden_size))) 
Win[1:,:] = (np.random.randint(-1, 2, size = (in_size,hidden_size))) 


Wout = np.random.randint(-2, 2, size = (1+hidden_size,out_size)).astype(np.float64)

def predict(X):
    A_out = np.where((np.dot(X, Win[1:, :]) + Win[0,:]) > 0.0, 1, 0).astype(np.float64)
    R_out = np.where((np.dot(A_out, Wout[1:, :]) + Wout[0,:]) > 0.0, 1, 0).astype(np.float64)
    return A_out, R_out

def train(X, Y, n_iter = 5):
    Wout_list = []
    Wout_list.append(Wout.copy())
    for i in range(n_iter):
        err = 0
        for Xi, target in zip(X, Y):
            hidden_out, pred = predict(Xi)
            Wout[1:] += (target - pred) * hidden_out.reshape(-1,1)
            Wout[0] += (target - pred).astype(np.float64)
            if((target - pred) != 0):
                err += 1  
        print('Эпоха: ' + str(i) + ' | | ' + 'Количество ошибок: ' +  str(err))
        if(err == 0):
            print("Обучение завершено\nТекущая эпоха: " + str(i) + ' Текущие значения весов: ' + str(Wout[0:5, :]))
            break
        Wout_list.append(Wout.copy())
        if(np.all(Wout_list[-1] == Wout_list[-2])):
            print('Перцептрон зациклился\n текущая эпоха: ' + str(i) + ' Текущие веса: ' + str(Wout[0:5, :]))
            break

train(X,Y)
_, pred = predict(X)
