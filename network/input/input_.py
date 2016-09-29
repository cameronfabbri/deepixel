import tensorflow as tf
import sys
import cv2

sys.path.insert(0, '../utils/')
import config

data_dir = config.data_dir
dataset  = config.dataset
num_classes = config.num_classes

def read_and_decode(filename_queue):

   reader = tf.TFRecordReader()
   _, serialized_example = reader.read(filename_queue)
   features = tf.parse_single_example(
      serialized_example,
      features={
         'image': tf.FixedLenFeature([], tf.string),
      }
   )

   image = tf.decode_raw(features['image'], tf.uint8)
   image = tf.to_float(image, name='float32')
   
   image = tf.reshape(image, [128,128,3])
   # do some distortions here later

   return image


def inputs(type_input, batch_size):
   if type_input == "train":
      filename = data_dir+"/"+dataset+"/records/train.tfrecord"
   elif type_input == "test":
      filename = data_dir+"/"+dataset+"/records/test.tfrecord"

   filename_queue = tf.train.string_input_producer([filename])

   image = read_and_decode(filename_queue)
   '''
   images = tf.train.shuffle_batch([image], 
      batch_size=batch_size, 
      num_threads=5,
      capacity=1000+3*batch_size, 
      min_after_dequeue=1000)
   '''
   images = tf.train.batch([image],
      batch_size=batch_size,
      num_threads=2)
   return images

