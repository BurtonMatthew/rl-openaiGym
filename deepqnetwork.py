import tensorflow as tf

class DeepQNetwork:
    def __init__(self, numOutputs):
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(8,8), strides=4, padding='valid', activation='relu', input_shape=(84,84,4))) #"rectifier nonlinearity"
        self.model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(4,4), strides=2, padding='valid', activation='relu')) #"rectifier nonlinearity"
        self.model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=1, padding='valid', activation='relu')) #"rectifier"
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(512, activation='relu'))
        self.model.add(tf.keras.layers.Dense(numOutputs, activation='softmax'))
        #self.model.summary()