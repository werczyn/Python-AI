import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
import math
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

a = np.loadtxt('dane.txt')

yk = a[1:60, [0]]
yk1 = a[1:60, [1]]
yk2 = a[1:60, [2]]
yk3 = a[1:60, [3]]
x = a[1:60, [4]]

c = np.hstack([yk1, yk2, yk3, x])  # model ARX
v = np.linalg.pinv(c) @ yk

e1_train = (yk - v[0] * yk1 + v[1] * yk2 + v[2] * yk3 + v[3] * x) ** 2
e1_train = sum(e1_train) / len(e1_train)
print('wynik: ', e1_train)

plt.plot(yk, 'r-')
plt.plot(v[0] * yk1 + v[1] * yk2 + v[2] * yk3 + v[3] * x)
plt.savefig('plt1')
plt.show()

# !!! test !!!


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)



# fix random seed for reproducibility
np.random.seed(7)

# load the dataset
dataset = a[:, [0]]
dataset = dataset.astype('float32')

dataset = dataset[1:60, :]

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

# reshape into X=t and Y=t+1
look_back = 4
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# create and fit the LSTM network

model = Sequential()

model.add(LSTM(10, input_shape=(1, look_back)))
model.add(Dense(1))

'''
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))  
model.add(Dropout(0.2))

model.add(LSTM(units=50))  
model.add(Dropout(0.2))  
model.add(Dense(1))
'''

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
print('Test Score: %.2f RMSE' % (testScore))

# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict

# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(dataset) - 1, :] = testPredict

# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset), 'r-')
plt.plot(trainPredictPlot, "b-")
plt.plot(testPredictPlot, "g-")
plt.savefig('plt2')
plt.show()

