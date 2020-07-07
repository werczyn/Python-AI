from keras import layers
from keras import models
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

#budowa sieci
model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform'))
model.add(layers.Conv2DTranspose(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
model.add(layers.Conv2DTranspose(32, (3, 3), activation='relu', kernel_initializer='he_uniform'))
model.add(layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same'))
model.summary()

#wczytanie obrazów
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

# #uczenie sieci
# model.compile(optimizer='adam', loss='binary_crossentropy')
# model.fit(train_images, train_images, epochs=3, batch_size=64)
#
# #zapis sieci
# model.save('autocoder.h5')

# wczytanie sieci
model = models.load_model('autocoder.h5')
model.summary()

# szum
clear_train = train_images
clear_test = test_images
noise_train = clear_train + 0.35 * np.random.normal(0, 1, clear_train.shape)
noise_test = clear_test + 0.35 * np.random.normal(0, 1, clear_test.shape)

images_to_show = 5
predicted_images = model.predict(noise_test[:images_to_show])

for i in range(images_to_show):
    noise_image = noise_test[i][:, :, 0]
    clear_image = clear_test[i][:, :, 0]
    predict_image = predicted_images[i][:, :, 0]

    fig, ax = plt.subplots(1, 3)
    fig.set_size_inches(8, 3.5)
    ax[0].imshow(noise_image)
    ax[1].imshow(clear_image)
    ax[2].imshow(predict_image)
    ax[0].set_title('noisy image')
    ax[1].set_title('clear image')
    ax[2].set_title('predicted image')
    plt.show()