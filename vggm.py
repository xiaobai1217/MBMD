import tensorflow as tf


class vggM:
    def __init__(self):
        self.learningRates = {}

    def extractFeature(self, inputs):
        with tf.variable_scope('VGGM') as scope:
            scope.reuse_variables()
            with tf.variable_scope('layer1'):
                weights = tf.get_variable("weights")
                biases = tf.get_variable("biases")
                outputs = tf.nn.conv2d(inputs, weights, strides=[1, 2, 2, 1], padding='VALID') + biases
                outputs = tf.nn.relu(outputs)
                outputs = tf.nn.lrn(outputs, depth_radius=2, bias=1.0, alpha=0.0001, beta=0.75)
                outputs = tf.nn.max_pool(outputs, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

            with tf.variable_scope('layer2'):
                weights = tf.get_variable("weights")
                biases = tf.get_variable("biases")
                outputs = tf.nn.conv2d(outputs, weights, strides=[1, 2, 2, 1], padding='VALID') + biases
                outputs = tf.nn.relu(outputs)
                outputs = tf.nn.lrn(outputs, depth_radius=2, bias=1.0, alpha=0.0001, beta=0.75)
                outputs = tf.nn.max_pool(outputs, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

            with tf.variable_scope('layer3'):
                weights = tf.get_variable("weights")
                biases = tf.get_variable("biases")
                outputs3 = tf.nn.conv2d(outputs, weights, strides=[1, 1, 1, 1], padding='VALID') + biases
                outputs = tf.nn.relu(outputs3)

        return outputs

    def classification(self,inputs):
        with tf.variable_scope('VGGM') as scope:
            scope.reuse_variables()
            with tf.variable_scope('layer4'):
                weights = tf.get_variable("weights")
                biases = tf.get_variable("biases")
                outputs = tf.nn.dropout(inputs, keep_prob=0.5)
                outputs = tf.nn.conv2d(outputs, weights, strides=[1, 1, 1, 1], padding='VALID') + biases
                outputs = tf.nn.relu(outputs)

            with tf.variable_scope('layer5'):
                outputs = tf.contrib.layers.flatten(outputs)
                weights = tf.get_variable("weights")
                biases = tf.get_variable("biases")
                outputs = tf.nn.dropout(outputs, keep_prob=0.5)
                outputs = tf.matmul(outputs, weights) + biases
                outputs = tf.nn.relu(outputs)

            with tf.variable_scope('layer6'):
                weights = tf.get_variable("weights")
                biases = tf.get_variable("biases")
                outputs = tf.nn.dropout(outputs, keep_prob=0.5)
                outputs = tf.matmul(outputs, weights) + biases
        score = tf.nn.softmax(outputs,dim = 1)
        return outputs,score

    def loss(self,inputs,label):
        loss1 = tf.losses.softmax_cross_entropy(onehot_labels=label,logits=inputs)
        loss = tf.reduce_sum(loss1)
        #with tf.variable_scope('VGGM'):
        #    regularization = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
#        loss = tf.add_n([loss]+regularization)
        with tf.variable_scope("VGGM") as scope:
            scope.reuse_variables()
            with tf.variable_scope("layer4"):
                weights1 = tf.get_variable("weights")
            with tf.variable_scope("layer5"):
                weights2 = tf.get_variable("weights")
            with tf.variable_scope("layer6"):
                weights3 = tf.get_variable("weights")

        loss += (tf.nn.l2_loss(weights1)+ tf.nn.l2_loss(weights2) + tf.nn.l2_loss(weights3)) * 5e-4
        return loss,loss1

    def vggM(self,inputs,reuse=False):
        with tf.variable_scope('VGGM') as scope:
            if reuse is True:
                scope.reuse_variables()
            with tf.variable_scope('layer1'):
                weights = tf.get_variable("weights", shape=(7, 7, 3, 96), dtype=tf.float32,
                                          initializer=tf.truncated_normal_initializer(0.01), trainable=False)
                biases = tf.get_variable("biases", shape=(96,), dtype=tf.float32, initializer=tf.constant_initializer(0.0),trainable=False)
                outputs = tf.nn.conv2d(inputs, weights, strides=[1, 2, 2, 1], padding='VALID') + biases
                outputs = tf.nn.relu(outputs)
                outputs = tf.nn.lrn(outputs, depth_radius=2, bias=1.0, alpha=0.0001, beta=0.75)
                outputs = tf.nn.max_pool(outputs, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

            with tf.variable_scope('layer2'):
                weights = tf.get_variable("weights", shape=(5, 5, 96, 256), dtype=tf.float32,
                                          initializer=tf.truncated_normal_initializer(0.01), trainable=False)
                biases = tf.get_variable("biases", shape=(256,), dtype=tf.float32, initializer=tf.constant_initializer(0.0), trainable=False)
                outputs = tf.nn.conv2d(outputs, weights, strides=[1, 2, 2, 1], padding='VALID') + biases
                outputs = tf.nn.relu(outputs)
                outputs = tf.nn.lrn(outputs, depth_radius=2, bias=1.0, alpha=0.0001, beta=0.75)
                outputs = tf.nn.max_pool(outputs, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

            with tf.variable_scope('layer3'):
                weights = tf.get_variable("weights", shape=(3, 3, 256, 512), dtype=tf.float32,
                                          initializer=tf.truncated_normal_initializer(0.01), trainable=False)
                biases = tf.get_variable("biases", shape=(512,), dtype=tf.float32, initializer=tf.constant_initializer(0.0),trainable=False)
                outputs3 = tf.nn.conv2d(outputs, weights, strides=[1, 1, 1, 1], padding='VALID') + biases
                outputs = tf.nn.relu(outputs3)

            with tf.variable_scope('layer4'):
                weights = tf.get_variable("weights", shape=(3, 3, 512, 512), dtype=tf.float32,
                                          initializer=tf.truncated_normal_initializer(0.01),
                                          regularizer=tf.contrib.layers.l2_regularizer(5e-4), trainable=True)
                biases = tf.get_variable("biases", shape=(512,), dtype=tf.float32, initializer=tf.constant_initializer(0.0),
                                         regularizer=tf.contrib.layers.l2_regularizer(5e-4), trainable=True)
                #outputs = tf.nn.dropout(outputs,keep_prob=0.5)
                outputs = tf.nn.conv2d(outputs, weights, strides=[1, 1, 1, 1], padding='VALID') + biases
                outputs = tf.nn.relu(outputs)

            with tf.variable_scope('layer5'):
                outputs = tf.contrib.layers.flatten(outputs)
                weights = tf.get_variable("weights", shape=(512, 512), dtype=tf.float32,
                                          initializer=tf.contrib.layers.xavier_initializer(),
                                          regularizer=tf.contrib.layers.l2_regularizer(5e-4), trainable=True)
                biases = tf.get_variable("biases", shape=(512,), dtype=tf.float32, initializer=tf.constant_initializer(0.0),
                                         regularizer=tf.contrib.layers.l2_regularizer(5e-4), trainable=True)
                #outputs = tf.nn.dropout(outputs,keep_prob=0.5)
                outputs = tf.matmul(outputs,weights) + biases
                outputs = tf.nn.relu(outputs)

            with tf.variable_scope('layer6'):
                weights = tf.get_variable("weights", shape=(512, 2), dtype=tf.float32,
                                          initializer=tf.contrib.layers.xavier_initializer(),
                                          regularizer=tf.contrib.layers.l2_regularizer(5e-4), trainable=True)
                biases = tf.get_variable("biases", shape=(2,), dtype=tf.float32, initializer=tf.constant_initializer(0.0),
                                         regularizer=tf.contrib.layers.l2_regularizer(5e-4), trainable=True)
                #outputs = tf.nn.dropout(outputs,keep_prob=0.5)
                outputs = tf.matmul(outputs,weights)+ biases
        return outputs