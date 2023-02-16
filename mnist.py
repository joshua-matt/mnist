import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import layers

train, test = tf.keras.datasets.mnist.load_data()
x_train, y_train = train[0], tf.one_hot(train[1], 10)
x_test, y_test = test[0], tf.one_hot(test[1], 10)

model = keras.Sequential([
    layers.Flatten(),
    layers.Dense(16, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["categorical_accuracy"]
)

history = model.fit(x_train, y_train,
                    validation_data=(x_test, y_test),
                    epochs=10,
                    batch_size=32)
history_frame = pd.DataFrame(history.history)
history_frame.loc[:, ['loss', 'val_loss']].plot()
history_frame.loc[:, ['categorical_accuracy', 'val_categorical_accuracy']].plot()

i = np.random.choice(100)
print(x_test[i])
#print(model.prdict(x_test[i])) TODO make it predict!
plt.imshow(x_test[i], cmap='gray', vmin=0, vmax=255)
plt.show()

plt.show()