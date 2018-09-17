import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages

MOVING_AVERAGE_DECAY = 0.9997
UPDATE_OPS_COLLECTION = 'sf_update_ops'

class SiameseNet:
    learningRates = None

    def __init__(self):
        self.learningRates = {}

    def buildExemplarSubNetwork(self, exemplar, opts, isTrainingOp, branchType="original"):
        with tf.variable_scope('siamese') as scope:
            scope.reuse_variables()
            score = self.buildBranch(exemplar, opts, isTrainingOp, branchType=branchType)

        return score

    def buildInferenceNetwork(self, instance, zFeat, opts, isTrainingOp, branchType="original"):

        with tf.variable_scope('siamese') as scope:
            scope.reuse_variables()
            score = self.buildBranch(instance, opts, isTrainingOp, branchType=branchType)

        with tf.variable_scope('score'):
            batchAFeat = int(zFeat.get_shape()[-1])
            batchScore = int(score.get_shape()[0])

            assert batchAFeat == 1
            assert batchScore == opts['numScale']

            scores = tf.split(axis=0, num_or_size_splits=batchScore, value=score)
            scores1 = []
            for i in range(batchScore):
                scores1.append(tf.nn.conv2d(scores[i], zFeat, strides=[1, 1, 1, 1], padding='VALID'))

            score = tf.concat(axis=0, values=scores1)

        with tf.variable_scope('adjust') as scope:
            scope.reuse_variables()
            print("Building adjust...")
            weights = self.getVariable('weights', [1, 1, 1, 1],
                                       initializer=tf.constant_initializer(value=0.001, dtype=tf.float32),
                                       weightDecay=1.0 * opts['trainWeightDecay'], dType=tf.float32, trainable=True)
            # tf.get_variable('weights', [1, 1, 1, 1], initializer=tf.constant_initializer(value=0.001, dtype=tf.float32))
            biases = self.getVariable('biases', [1, ],
                                      initializer=tf.constant_initializer(value=0, dtype=tf.float32),
                                      weightDecay=1.0 * opts['trainWeightDecay'], dType=tf.float32, trainable=True)
            # tf.get_variable('biases', [1, ], initializer=tf.constant_initializer(value=0, dtype=tf.float32))
            score = tf.nn.conv2d(score, weights, strides=[1, 1, 1, 1], padding='VALID')
            score = tf.add(score, biases)

        return score

    def buildTrainNetwork(self, exemplar, instance, opts, isTraining=True, branchType="original"):
        params = {}
        params['trainBatchSize'] = exemplar.get_shape()[0]
        isTrainingOp = tf.convert_to_tensor(isTraining, dtype='bool', name='is_training')

        with tf.variable_scope('siamese') as scope:
            aFeat = self.buildBranch(exemplar, opts, isTrainingOp, branchType=branchType) #, name='aFeat'
            scope.reuse_variables()
            score = self.buildBranch(instance, opts, isTrainingOp, branchType=branchType) #, name='xFeat'

            # the conv2d op in tf is used to implement xcorr directly, from theory, the implementation of conv2d is correlation. However, it is necessary to transpose the weights tensor to a input tensor
            # different scales are tackled with slicing the data. Now only 3 scales are considered, but in training, more samples in a batch is also tackled by the same mechanism. Hence more slices is to be implemented here!!
        with tf.variable_scope('score'):
            print("Building xcorr...")
            aFeat = tf.transpose(aFeat, perm=[1, 2, 3, 0])
            batchAFeat = int(aFeat.get_shape()[-1])
            batchScore = int(score.get_shape()[0])

            # if batchAFeat > 1:
            groupConv = lambda i, k: tf.nn.conv2d(i, k, strides=[1, 1, 1, 1], padding='VALID')

            #assert batchAFeat == params['trainBatchSize']
            #assert batchScore == params['trainBatchSize']

            aFeats = tf.split(axis=3, num_or_size_splits=batchAFeat, value=aFeat)
            scores = tf.split(axis=0, num_or_size_splits=batchScore, value=score)
            scores = [groupConv(i, k) for i, k in zip(scores, aFeats)]

            score = tf.concat(axis=3, values=scores)
            score = tf.transpose(score, perm=[3, 1, 2, 0])
            # else:

        with tf.variable_scope('adjust'):
            print("Building adjust...")
            weights = self.getVariable('weights', [1, 1, 1, 1], initializer=tf.constant_initializer(value=0.001, dtype=tf.float32), weightDecay=1.0*opts['trainWeightDecay'], dType=tf.float32, trainable=True)
            self.learningRates[weights.name] = 0.0
                # tf.get_variable('weights', [1, 1, 1, 1], initializer=tf.constant_initializer(value=0.001, dtype=tf.float32))
            biases = self.getVariable('biases', [1,], initializer=tf.constant_initializer(value=0, dtype=tf.float32), weightDecay=1.0*opts['trainWeightDecay'], dType=tf.float32, trainable=True)
            self.learningRates[biases.name] = 1.0
                # tf.get_variable('biases', [1, ], initializer=tf.constant_initializer(value=0, dtype=tf.float32))
            score = tf.nn.conv2d(score, weights, strides=[1, 1, 1, 1], padding='VALID')
            score = tf.add(score, biases)

        return score

    def buildBranch(self, inputs, opts, isTrainingOp, branchType="original", branchName=None):
        if branchType == "original":
            return self.buildOriBranch(inputs, opts, isTrainingOp, branchName)
        elif branchType == "simple":
            return self.buildSimpleBranch(inputs, opts, isTrainingOp, branchName)
        else:
            return

    def buildSimpleBranch(self, inputs, opts, isTrainingOp, branchName):
        print("Building Siamese branches...")

        with tf.variable_scope('scala1'):
            print("Building conv1, bn1, relu1, pooling1...")
            name = tf.get_variable_scope().name
            # outputs = conv1(inputs, 3, 96, 11, 2)
            outputs = self.conv(inputs, 96, 11, 2, 1, [1.0, 2.0], [1.0, 0.0], opts['trainWeightDecay'], opts['stddev'])
            outputs = self.batchNorm(outputs, isTrainingOp) #batchNormalization(outputs, isTrainingOp, name)
            outputs = tf.nn.relu(outputs)
            outputs = self.maxPool(outputs, 3, 2)

        with tf.variable_scope('scala2'):
            print("Building conv2, bn2, relu2, pooling2...")
            name = tf.get_variable_scope().name
            # outputs = conv2(outputs, 48, 256, 5, 1)
            outputs = self.conv(outputs, 256, 5, 1, 1, [1.0, 2.0], [1.0, 0.0], opts['trainWeightDecay'], opts['stddev'])
            outputs = self.batchNorm(outputs, isTrainingOp)
            outputs = tf.nn.relu(outputs)
            outputs = self.maxPool(outputs, 3, 2)

        with tf.variable_scope('scala3'):
            print("Building conv3, bn3, relu3...")
            name = tf.get_variable_scope().name
            # outputs = conv1(outputs, 256, 384, 3, 1)
            outputs = self.conv(outputs, 384, 3, 1, 1, [1.0, 2.0], [1.0, 0.0], opts['trainWeightDecay'], opts['stddev'])
            outputs = self.batchNorm(outputs, isTrainingOp)
            outputs = tf.nn.relu(outputs)

        with tf.variable_scope('scala4'):
            print("Building conv4, bn4, relu4...")
            name = tf.get_variable_scope().name
            # outputs = conv2(outputs, 192, 384, 3, 1)
            outputs = self.conv(outputs, 384, 3, 1, 1, [1.0, 2.0], [1.0, 0.0], opts['trainWeightDecay'], opts['stddev'])
            outputs = self.batchNorm(outputs, isTrainingOp)
            outputs = tf.nn.relu(outputs)

        with tf.variable_scope('scala5'):
            print("Building conv5...")
            # outputs = conv2(outputs, 192, 256, 3, 1)
            outputs = self.conv(outputs, 256, 3, 1, 1, [1.0, 2.0], [1.0, 0.0], opts['trainWeightDecay'], opts['stddev'], name=branchName)

        return outputs

    def buildOriBranch(self, inputs, opts, isTrainingOp, branchName):
        print("Building Siamese branches...")

        with tf.variable_scope('scala1'):
            print("Building conv1, bn1, relu1, pooling1...")
            name = tf.get_variable_scope().name
            # outputs = conv1(inputs, 3, 96, 11, 2)
            outputs = self.conv(inputs, 96, 11, 2, 1, [1.0, 2.0], [1.0, 0.0], opts['trainWeightDecay'], opts['stddev'])
            outputs = self.batchNorm(outputs, isTrainingOp)
            outputs = tf.nn.relu(outputs)
            outputs = self.maxPool(outputs, 3, 2)

        with tf.variable_scope('scala2'):
            print("Building conv2, bn2, relu2, pooling2...")
            name = tf.get_variable_scope().name
            # outputs = conv2(outputs, 48, 256, 5, 1)
            outputs = self.conv(outputs, 256, 5, 1, 2, [1.0, 2.0], [1.0, 0.0], opts['trainWeightDecay'], opts['stddev'])
            outputs = self.batchNorm(outputs, isTrainingOp)
            outputs = tf.nn.relu(outputs)
            outputs = self.maxPool(outputs, 3, 2)

        with tf.variable_scope('scala3'):
            print("Building conv3, bn3, relu3...")
            name = tf.get_variable_scope().name
            # outputs = conv1(outputs, 256, 384, 3, 1)
            outputs3 = self.conv(outputs, 384, 3, 1, 1, [1.0, 2.0], [1.0, 0.0], opts['trainWeightDecay'], opts['stddev'])
            outputs = self.batchNorm(outputs3, isTrainingOp)
            outputs = tf.nn.relu(outputs)

        with tf.variable_scope('scala4'):
            print("Building conv4, bn4, relu4...")
            name = tf.get_variable_scope().name
            # outputs = conv2(outputs, 192, 384, 3, 1)
            outputs = self.conv(outputs, 384, 3, 1, 2, [1.0, 2.0], [1.0, 0.0], opts['trainWeightDecay'], opts['stddev'])
            outputs = self.batchNorm(outputs, isTrainingOp)
            outputs = tf.nn.relu(outputs)

        with tf.variable_scope('scala5'):
            print("Building conv5...")
            # outputs = conv2(outputs, 192, 256, 3, 1)
            outputs = self.conv(outputs, 256, 3, 1, 2, [1.0, 2.0], [1.0, 0.0], opts['trainWeightDecay'], opts['stddev'], name=branchName)

        return outputs

    def conv(self, inputs, filters, size, stride, groups, lrs, wds, wd, stddev, name=None):
        channels = int(inputs.get_shape()[-1])
        groupConv = lambda i, k: tf.nn.conv2d(i, k, strides=[1, stride, stride, 1], padding='VALID')

        with tf.variable_scope('conv'):
            weights = self.getVariable('weights', shape=[size, size, channels / groups, filters], initializer=tf.truncated_normal_initializer(stddev=stddev), weightDecay=wds[0]*wd, dType=tf.float32, trainable=True)
            # tf.get_variable('weights', shape=[size, size, channels/groups, filters], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32) ,
            biases = self.getVariable('biases', shape=[filters, ], initializer=tf.constant_initializer(value=0.1, dtype=tf.float32), weightDecay=wds[1]*wd, dType=tf.float32, trainable=True)
            # tf.get_variable('biases', [filters,], initializer=tf.constant_initializer(value=0.1, dtype=tf.float32))

        self.learningRates[weights.name] = lrs[0]
        self.learningRates[biases.name] = lrs[1]

        if groups == 1:
            conv = groupConv(inputs, weights)
        else:
            inputGroups = tf.split(axis=3, num_or_size_splits=groups, value=inputs)
            weightsGroups = tf.split(axis=3, num_or_size_splits=groups, value=weights)
            convGroups = [groupConv(i, k) for i, k in zip(inputGroups, weightsGroups)]

            conv = tf.concat(axis=3, values=convGroups)

        if name is not None:
            conv = tf.add(conv, biases, name=name)
        else:
            conv = tf.add(conv, biases)

        print('Layer Type = Conv, Size = %d * %d, Stride = %d, Filters = %d, Input channels = %d, Groups = %d' % (size, size, stride, filters, channels, groups))

        return conv

    def batchNorm(self, x, isTraining):
        shape = x.get_shape()
        paramsShape = shape[-1:]

        axis = list(range(len(shape)-1))

        with tf.variable_scope('bn'):
            beta = self.getVariable('beta', paramsShape, initializer=tf.constant_initializer(value=0, dtype=tf.float32))
            self.learningRates[beta.name] = 1.0
            gamma = self.getVariable('gamma', paramsShape, initializer=tf.constant_initializer(value=1, dtype=tf.float32))
            self.learningRates[gamma.name] = 2.0
            movingMean = self.getVariable('moving_mean', paramsShape, initializer=tf.constant_initializer(value=0, dtype=tf.float32), trainable=False)
            movingVariance = self.getVariable('moving_variance', paramsShape, initializer=tf.constant_initializer(value=1, dtype=tf.float32), trainable=False)

        mean, variance = tf.nn.moments(x, axis)
        updateMovingMean = moving_averages.assign_moving_average(movingMean, mean, MOVING_AVERAGE_DECAY)
        updateMovingVariance = moving_averages.assign_moving_average(movingVariance, variance, MOVING_AVERAGE_DECAY)
        tf.add_to_collection(UPDATE_OPS_COLLECTION, updateMovingMean)
        tf.add_to_collection(UPDATE_OPS_COLLECTION, updateMovingVariance)

        mean, variance = control_flow_ops.cond(isTraining, lambda : (mean, variance), lambda : (movingMean, movingVariance))

        x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, variance_epsilon=0.001)

        return x

    # def batchNormalization(self, inputs, isTraining, name):
    #     with tf.variable_scope('bn'):
    #         output = tf.contrib.layers.batch_norm(inputs, center=True, scale=True, is_training=isTraining, decay=0.997, epsilon=0.0001)
    #     self.learningRates[name+'/bn/BatchNorm/gamma:0'] = 2.0
    #     self.learningRates[name+'/bn/BatchNorm/beta:0'] = 1.0
    #
    #     return output

    def maxPool(self, inputs, kSize, _stride):
        with tf.variable_scope('poll'):
            output = tf.nn.max_pool(inputs, ksize=[1, kSize, kSize, 1], strides=[1, _stride, _stride, 1], padding='VALID')

        return output

    # the code here is strictly analogous to the matlab version of siamese-fc, weighted logistic loss function
    # however, weighted cross entropy loss can also be used with tf implementation
    def loss(self, score, y, weights):
        a = -tf.multiply(score, y)
        b = tf.nn.relu(a)
        loss = b+tf.log(tf.exp(-b)+tf.exp(a-b))
        # loss = tf.log(1+tf.exp(a))
        # loss = tf.reduce_mean(loss)
        loss = tf.reduce_mean(tf.multiply(weights, loss))
        regularization = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss = tf.add_n([loss]+regularization)

        return loss

    def getVariable(self, name, shape, initializer, weightDecay = 0.0, dType=tf.float32, trainable = True):
        if weightDecay > 0:
            regularizer = tf.contrib.layers.l2_regularizer(weightDecay)
        else:
            regularizer = None

        return tf.get_variable(name, shape=shape, initializer=initializer, dtype=dType, regularizer=regularizer, trainable=trainable)

# deprecated
def conv1(inputs, channels, filters, size, stride):
    # initializations include trancated norm distribution method and xavier method, the matlab version exploits an improved xavier method.
    # However I didn't find it in tf, so xavier is used here, if not work, something may need change here!!
    weights = tf.get_variable('weights', [size, size, channels, filters],
                              initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
    biases = tf.get_variable('biases', [filters, ],
                             initializer=tf.constant_initializer(value=0.1, dtype=tf.float32))

    conv = tf.nn.conv2d(inputs, weights, strides=[1, stride, stride, 1], padding='VALID')
    conv = tf.add(conv, biases)
    print('Layer Type = Conv, Size = %d * %d, Stride = %d, Filters = %d, Input channels = %d' % (
        size, size, stride, filters, channels))

    return conv

    # deprecated
def conv2(inputs, channels, filters, size, stride):
    inputShape = inputs.get_shape()

    inputs0 = tf.slice(inputs, [0, 0, 0, 0], [inputShape[0], inputShape[1], inputShape[2], channels])
    inputs1 = tf.slice(inputs, [0, 0, 0, channels], [inputShape[0], inputShape[1], inputShape[2], channels])

    weights0 = tf.get_variable('weights0', [size, size, channels, filters / 2],
                               initializer=tf.contrib.layers.xavier_initializer_conv2d(), dtype=tf.float32)
    weights1 = tf.get_variable('weights1', [size, size, channels, filters / 2],
                               initializer=tf.contrib.layers.xavier_initializer_conv2d(), dtype=tf.float32)

    conv0 = tf.nn.conv2d(inputs0, weights0, strides=[1, stride, stride, 1])
    conv1 = tf.nn.conv2d(inputs1, weights1, strides=[1, stride, stride, 1])
    conv = tf.concat([conv0, conv1], 3)

    biases = tf.get_variable('biases', [filters, ],
                             initializer=tf.constant_initializer(value=0.1, dtype=tf.float32))
    conv = tf.add(conv, biases)
    print('Layer Type = Conv, Size = %d * %d, Stride = %d, Filters = %d, Input channels = %d' % (
        size, size, stride, filters, channels))

    return conv