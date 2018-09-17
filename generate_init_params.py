import tensorflow as tf
import sys
import numpy as np
sys.path.append("/home/xiaobai/caffe/python")
import caffe
from vggm import vggM


lr = tf.placeholder(tf.float32, shape=())

net_caffe = caffe.Net("./VGGM.prototxt", "./VGG_CNN_M.caffemodel", caffe.TEST)
caffe_layers = {}
for i, layer in enumerate(net_caffe.layers):
    layer_name = net_caffe._layer_names[i]
    caffe_layers[layer_name] = layer

def caffe_weights(layer_name):
    layer = caffe_layers[layer_name]
    return layer.blobs[0].data

def caffe_bias(layer_name):
    layer = caffe_layers[layer_name]
    return layer.blobs[1].data

conv1_w = caffe_layers['conv1'].blobs[0].data.transpose((2, 3, 1, 0))
conv2_w = caffe_layers['conv2'].blobs[0].data.transpose((2, 3, 1, 0))
conv3_w = caffe_layers['conv3'].blobs[0].data.transpose((2, 3, 1, 0))
conv4_w = caffe_layers['conv4'].blobs[0].data.transpose((2, 3, 1, 0))
conv5_w = caffe_layers['conv5'].blobs[0].data.transpose((2, 3, 1, 0))
fc6_w = caffe_layers['fc6'].blobs[0].data.transpose((1,0))
fc7_w = caffe_layers['fc7'].blobs[0].data.transpose((1,0))

conv1_b = caffe_layers['conv1'].blobs[1].data
conv2_b = caffe_layers['conv2'].blobs[1].data
conv3_b = caffe_layers['conv3'].blobs[1].data
conv4_b = caffe_layers['conv4'].blobs[1].data
conv5_b = caffe_layers['conv5'].blobs[1].data
fc6_b = caffe_layers['fc6'].blobs[1].data
fc7_b = caffe_layers['fc7'].blobs[1].data

parameters_dict = {}
parameters_dict['conv1_w'] = conv1_w
parameters_dict['conv2_w'] = conv2_w
parameters_dict['conv3_w'] = conv3_w
parameters_dict['conv4_w'] = conv4_w
parameters_dict['conv5_w'] = conv5_w
parameters_dict['fc6_w'] = fc6_w
parameters_dict['fc7_w'] = fc7_w
parameters_dict['conv1_b'] = conv1_b
parameters_dict['conv2_b'] = conv2_b
parameters_dict['conv3_b'] = conv3_b
parameters_dict['conv4_b'] = conv4_b
parameters_dict['conv5_b'] = conv5_b
parameters_dict['fc6_b'] = fc6_b
parameters_dict['fc7_b'] = fc7_b

model = vggM()
inputs = tf.placeholder(dtype=tf.float32,shape=(128,107,107,3))
label = tf.placeholder(dtype=tf.uint8,shape=(128,2))
outputs = model.vggM(inputs)
lossOp = model.loss(outputs,label)
optimizer = tf.train.MomentumOptimizer(learning_rate=0.0002,momentum=0.9)
saver = tf.train.Saver(max_to_keep=40)
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

with tf.variable_scope('VGGM'):
    with tf.variable_scope("layer1") as scope:
        scope.reuse_variables()
        weights = tf.get_variable("weights")
        biases = tf.get_variable("biases")
        sess.run(weights.assign(parameters_dict["conv1_w"]))
        sess.run(biases.assign(parameters_dict["conv1_b"]))

    with tf.variable_scope("layer2") as scope:
        scope.reuse_variables()
        weights = tf.get_variable("weights")
        biases = tf.get_variable("biases")
        sess.run(weights.assign(parameters_dict["conv2_w"]))
        sess.run(biases.assign(parameters_dict["conv2_b"]))

    with tf.variable_scope("layer3") as scope:
        scope.reuse_variables()
        weights = tf.get_variable("weights")
        biases = tf.get_variable("biases")
        sess.run(weights.assign(parameters_dict["conv3_w"]))
        sess.run(biases.assign(parameters_dict["conv3_b"]))

    with tf.variable_scope("layer4") as scope:
        scope.reuse_variables()
        weights = tf.get_variable("weights")
        biases = tf.get_variable("biases")
        sess.run(weights.assign(parameters_dict["conv4_w"]))
        sess.run(biases.assign(parameters_dict["conv4_b"]))


saveRes = saver.save(sess, 'vggMParams.ckpt')