from PIL import Image
import numpy as np
import tensorflow as tf
import time

img_size = (1152, 1280)

def read_and_decode(filename_queue):
 reader = tf.TFRecordReader()
 _, serialized_example = reader.read(filename_queue)
 features = tf.parse_single_example(
  serialized_example,
  # Defaults are not specified since both keys are required.
  features={
      'hd_image': tf.FixedLenFeature([], tf.string),
      'img': tf.FixedLenFeature([], tf.string),
  })
 hd_image = tf.decode_raw(features['hd_image'], tf.uint8)
 img = tf.cast(features['img'], tf.int32)
 return hd_image, img 


def get_all_records(FILE):
 with tf.Session() as sess:
   filename_queue = tf.train.string_input_producer([ FILE ])
   hd_image, img = read_and_decode(filename_queue)
   hd_image = tf.reshape(hd_image, tf.pack([img_size[0], img_size[1], 3]))
   hd_image.set_shape([img_size[0],img_size[1],3])
   init_op = tf.initialize_all_variables()
   sess.run(init_op)
   coord = tf.train.Coordinator()
   threads = tf.train.start_queue_runners(coord=coord)
   for i in range(2053):
     t = time.time()
     example = sess.run([hd_image])[0]
     print(example)
     img = Image.fromarray(example, 'RGB')
     img.save( "output/" + str(i) + '-train.png')
     elapsed = time.time() - t
     print("time per batch is " + str(elapsed))

   coord.request_stop()
   coord.join(threads)

get_all_records('./test_images.tfrecord')
