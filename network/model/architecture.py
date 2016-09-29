import tensorflow as tf
import numpy as np
import sys
sys.path.insert(0, '../utils/')
import config

FLAGS = tf.app.flags.FLAGS

num_epochs = 100

tf.app.flags.DEFINE_integer('batch_size', config.batch_size,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_float('weight_decay', 0.0005,
                          """ """)
tf.app.flags.DEFINE_float('alpha', 0.1,
                          """Leaky RElu param""")

data_dir = config.data_dir
num_classes = config.num_classes

def _variable_on_cpu(name, shape, initializer):
   with tf.device('/cpu:0'):
      var = tf.get_variable(name, shape, initializer=initializer)
   return var


def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.
  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.
  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
  Returns:
    Variable Tensor
  """
  var = _variable_on_cpu(name, shape,
                         tf.truncated_normal_initializer(stddev=stddev))
  if wd:
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    weight_decay.set_shape([])
    tf.add_to_collection('losses', weight_decay)
  return var


def _conv_layer(inputs, kernel_size, stride, num_features, idx):
   with tf.variable_scope('{0}_conv'.format(idx)) as scope:
      input_channels = inputs.get_shape()[3]

      weights = _variable_with_weight_decay('weights', shape=[kernel_size, kernel_size, input_channels, num_features], stddev=0.1, wd=FLAGS.weight_decay)
      biases = _variable_on_cpu('biases', [num_features], tf.constant_initializer(0.1))

      conv = tf.nn.conv2d(inputs, weights, strides=[1, stride, stride, 1], padding='SAME')
      conv_biased = tf.nn.bias_add(conv, biases)

      #Leaky ReLU
      conv_rect = tf.maximum(FLAGS.alpha*conv_biased, conv_biased, name='{0}_conv'.format(idx))
      return conv_rect


def _fc_layer(inputs, hiddens, idx, flat = False, linear = False):
  with tf.variable_scope('fc{0}'.format(idx)) as scope:
    input_shape = inputs.get_shape().as_list()
    if flat:
      dim = input_shape[1]*input_shape[2]*input_shape[3]
      inputs_processed = tf.reshape(inputs, [-1,dim])
    else:
      dim = input_shape[1]
      inputs_processed = inputs

    weights = _variable_with_weight_decay('weights', shape=[dim,hiddens],stddev=0.01, wd=FLAGS.weight_decay)
    biases = _variable_on_cpu('biases', [hiddens], tf.constant_initializer(0.01))
    if linear:
      return tf.add(tf.matmul(inputs_processed,weights),biases,name=str(idx)+'_fc')

    ip = tf.add(tf.matmul(inputs_processed,weights),biases)
    return tf.maximum(FLAGS.alpha*ip,ip,name=str(idx)+'_fc')

def deconv(inputs, stride, out_shape, kernel_size, num_features, idx):
   with tf.variable_scope('fc{0}'.format(idx)) as scope:
      input_channels = inputs.get_shape()[3]

      filter_ = _variable_with_weight_decay('weights', shape=[kernel_size, kernel_size, num_features, input_channels], stddev=0.1, wd=FLAGS.weight_decay)
      strides=[1, stride, stride, 1]

      d_conv = tf.nn.conv2d_transpose(inputs, filter_, output_shape=out_shape, strides=strides, padding='SAME') 
      return tf.maximum(FLAGS.alpha*d_conv,d_conv,name=str(idx))

def inference(images, name):
           # input, kernel size, stride, num_features, num_epochs
   conv1 = tf.nn.dropout(images, .8)
   conv1 = _conv_layer(conv1, 7, 2, 32, '1')

   # perform a 20% dropout after the first layer

   conv2 = _conv_layer(conv1, 2, 2, 32, '2')

   conv3 = _conv_layer(conv2, 5, 2, 64, '3')
   
   fc4 = _fc_layer(conv3, 1024, '4', True, False)
   
   fc5 = _fc_layer(fc4, 512, '5', False, False)
  
   fc6 = _fc_layer(fc5, 256, '6', False, False)

   # convert last layer to image
   fc7 = _fc_layer(fc6, 512, '7', False, False)
   
   fc8 = _fc_layer(fc7, 1024, '8', False, False)
   
   fc9 = _fc_layer(fc8, 16*16*64, '9', False, False)
   
   # reshape fc9
   fc9 = tf.reshape(fc9, [config.batch_size, 16, 16, 64])

   # perform deconvolutions
   # output shape is the output shape of conv2
   out_shape = tf.pack([config.batch_size, 32, 32, 32])
   d_conv9 = deconv(fc9, 2, out_shape, 5, 32, '10')

   out_shape = tf.pack([config.batch_size, 64, 64, 32])
   d_conv10 = deconv(d_conv9, 2, out_shape, 2, 32, '11')

   out_shape = tf.pack([config.batch_size, 128, 128, 3])
   d_conv11 = deconv(d_conv10, 2, out_shape, 7, 3, '13')

   return d_conv11

def loss (input_images, logits):
   error = tf.nn.l2_loss(input_images - logits)
   return error 


