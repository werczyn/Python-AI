from keras import layers
from keras import models
from keras.datasets import cifar10
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np

model1 = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dense(3, activation='softmax')
])
model1.summary()
print("")

model2 = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dense(3, activation='softmax')
])
model2.summary()
print("")

model3 = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dense(3, activation='softmax')
])
model3.summary()
print("")

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x = np.concatenate((x_train, x_test))
y = np.concatenate((y_train, y_test))

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7)

y_train2 = y_train.copy()
y_train2[y_train2 > 1] = -1
y_test2 = y_test.copy()
y_test2[y_test2 > 1] = -1

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
y_train2[y_train2 == 255] = 2
y_test2[y_test2 == 255] = 2

y_train2 = to_categorical(y_train2, 3)
y_test2 = to_categorical(y_test2, 3)

model1.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model1.fit(x_train, y_train2, epochs=2, batch_size=64)
test_loss, test_acc1 = model1.evaluate(x_test, y_test2)


model2.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model2.fit(x_train, y_train2, epochs=2, batch_size=64)
test_loss, test_acc2 = model2.evaluate(x_test, y_test2)


model3.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model3.fit(x_train, y_train2, epochs=2, batch_size=64)
test_loss, test_acc3 = model3.evaluate(x_test, y_test2)

print('Model 1, dokładność: ', test_acc1*100)
print('Model 2, dokładność: ', test_acc2*100)
print('Model 3, dokładność: ', test_acc3*100)
