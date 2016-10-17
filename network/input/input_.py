import tensorflow as tf
import sys
import cv2

sys.path.insert(0, '../utils/')

def read_and_decode(filename_queue):

   reader = tf.TFRecordReader()
   _, serialized_example = reader.read(filename_queue)
   features = tf.parse_single_example(
      serialized_example,
      features={
         'hd_image1': tf.FixedLenFeature([], tf.string),
         'img': tf.FixedLenFeature([], tf.string),
      }
   )

   hd_image = tf.decode_raw(features['hd_image1'], tf.uint8)
   hd_image = tf.to_float(hd_image, name='float32')
   #hd_image = tf.reshape(hd_image, [576,640,3])
   hd_image = tf.reshape(hd_image, [640,576,3])
   #hd_image = hd_image/255.0
   
   img = tf.decode_raw(features['img'], tf.uint8)
   img = tf.to_float(img, name='float32')
   #img = tf.reshape(img, [144,160,3])
   img = tf.reshape(img, [160,144,3])
   img = tf.image.per_image_whitening(img)
   #img = img/255.0

   return img, hd_image


def inputs(record_file, batch_size, type_):
   print(record_file)
   filename_queue = tf.train.string_input_producer([record_file])

   img, hd_image = read_and_decode(filename_queue)

   imgs, hd_images = tf.train.shuffle_batch([img, hd_image], 
      batch_size=batch_size, 
      num_threads=5,
      capacity=100+3*batch_size, 
      min_after_dequeue=100)
   
   return imgs, hd_images

