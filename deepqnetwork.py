import numpy
import tensorflow as tf

class DeepQNetwork:
    def __init__(self, numOutputs, stackSize):
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(8,8), strides=4, padding='valid', activation='relu', input_shape=[84,84,stackSize], dtype=tf.float32))
        self.model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(4,4), strides=2, padding='valid', activation='relu'))
        self.model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=1, padding='valid', activation='relu'))
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(512, activation='relu'))
        self.model.add(tf.keras.layers.Dense(numOutputs))
        self.model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.RMSprop(lr=0.00025))
        self.model.summary()

    def predict(self, frameStack):
        preparedStack = (numpy.stack(frameStack, axis=2) / 255.0).astype(numpy.float32)
        return self.model.predict(tf.expand_dims(tf.convert_to_tensor(preparedStack), 0), steps=1)