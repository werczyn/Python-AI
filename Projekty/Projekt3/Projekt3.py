import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

dane = np.loadtxt('Dane/dane15.txt')
x = dane[:, [0]]
y = dane[:, [1]]

print('X:')
print(x)
print('Y:')
print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

print('Model liniowy:')
c1 = np.hstack([x_train, np.ones(x_train.shape)])
v1 = np.linalg.pinv(c1) @ y_train

e1_train = (y_train - v1[0] * x_train - v1[1]) ** 2
e1_train = sum(e1_train) / len(e1_train)
e1_test = (y_test - v1[0] * x_test - v1[1]) ** 2
e1_test = sum(e1_test) / len(e1_test)
print('train: ', e1_train)
print('test: ', e1_test)

print('Model zlozony:')
c2 = np.hstack([x_train ** 3, x_train ** 2, x_train, np.ones(x_train.shape)])
v2 = np.linalg.pinv(c2) @ y_train

e2_train = (y_train - (v2[0] * x_train ** 3 + v2[1] * x_train ** 2 + v2[2] * x_train + v2[3])) ** 2
e2_train = sum(e2_train) / len(e2_train)
e2_test = (y_test - (v2[0] * x_test ** 3 + v2[1] * x_test ** 2 + v2[2] * x_test + v2[3])) ** 2
e2_test = sum(e2_test) / len(e2_test)
print('train: ', e2_train)
print('test: ', e2_test)

print('Model zlozony2: ')
c3 = np.hstack([x_train ** 4, x_train ** 3, x_train ** 2, x_train, np.ones(x_train.shape)])
v3 = np.linalg.pinv(c3) @ y_train

e3_train = (y_train - (v3[0] * x_train ** 4 + v3[1] * x_train ** 3 + v3[2] * x_train ** 2 + v3[3] * x_train + v3[4])) ** 2
e3_train = sum(e3_train) / len(e3_train)
e3_test = (y_test - (v3[0] * x_test ** 4 + v3[1] * x_test ** 3 + v3[2] * x_test ** 2 + v3[3] * x_test + v3[4])) ** 2
e3_test = sum(e3_test) / len(e3_test)
print('train: ', e3_train)
print('test: ', e3_test)

plt.plot(x, y, 'ro')
plt.plot(x, v1[0] * x + v1[1])
plt.plot(x, v2[0] * x ** 3 + v2[1] * x ** 2 + v2[2] * x + v2[3])
plt.plot(x, v3[0] * x ** 4 + v3[1] * x ** 3 + v3[2] * x ** 2 + v3[3] * x + v3[4])

plt.show()
