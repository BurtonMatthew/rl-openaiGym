import numpy
import tensorflow as tf

class DeepQNetwork:
    def __init__(self, numOutputs, stackSize):
        #prep the inputs
        self.inputs = tf.placeholder(shape=[None,84,84,stackSize], dtype=tf.uint8)
        self.qys = tf.placeholder(shape=[None], dtype=tf.float32)
        self.selectedActions = tf.placeholder(shape=[None,numOutputs], dtype=tf.float32)

        floatInputs = tf.cast(self.inputs, dtype=tf.float32)
        normalizedInputs = tf.math.divide(floatInputs, 255)

        #first convolution layer
        conv1 = tf.nn.conv2d(input=normalizedInputs, filter=tf.Variable(tf.truncated_normal([8,8,stackSize,32], stddev=0.5)), strides=[1,4,4,1], padding="VALID")
        conv1Out = tf.nn.relu(conv1)

        #second convolution layer
        conv2 = tf.nn.conv2d(input=conv1Out, filter=tf.Variable(tf.truncated_normal([4,4,32,64], stddev=0.5)), strides=[1,3,3,1], padding="VALID")
        conv2Out = tf.nn.relu(conv2)

        #third convolution layer
        conv3 = tf.nn.conv2d(input=conv2Out, filter=tf.Variable(tf.truncated_normal([3,3,64,64], stddev=0.5)), strides=[1,2,2,1], padding="VALID")
        conv3Out = tf.nn.relu(conv3)

        #flatten
        #convFlattened = tf.reshape(conv3Out, [None,-1])
        convFlattened = tf.layers.flatten(conv3Out)

        #hidden layer
        hidden = tf.layers.dense(inputs=convFlattened, units=512, activation=tf.nn.relu)

        #output
        self.outputs = tf.layers.dense(inputs=hidden, units=numOutputs)

        #training
        q = tf.reduce_sum(tf.multiply(self.outputs, self.selectedActions), axis=1)
        loss = tf.reduce_mean(tf.square(self.qys - q))
        self.trainFn = tf.train.RMSPropOptimizer(0.00025).minimize(loss)

    def predict(self, session, frameStack):
        return session.run(self.outputs, feed_dict = { self.inputs: numpy.expand_dims(numpy.stack(frameStack, axis=2), axis=0) })

    def train(self, session, batch):
        xs = []
        ys = []
        for xObs, _, _, _, yObs in batch:
            xs.append(numpy.stack(xObs, axis=2))
            ys.append(numpy.stack(yObs, axis=2))

        yPredicts = session.run(self.outputs, feed_dict = { self.inputs: numpy.stack(ys) })

        idx = 0
        qys = []
        actions = []
        for _, action, reward, done, _ in batch:
            if done:
                qys.append(reward)
            else:
                qys.append(reward + .99 * max(yPredicts[idx]))
            newAction = [0,0,0,0,0,0]
            newAction[action] = 1
            actions.append(newAction)
            idx += 1

        session.run(self.trainFn, feed_dict = {self.inputs: numpy.stack(xs), self.qys: numpy.stack(qys), self.selectedActions: numpy.stack(actions)})

    def save(self):
        return 0
    #    self.model.save("model.h5")