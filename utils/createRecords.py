"""
Cameron Fabbri

Creating three records, one train, one test, and one val, from the bird dataset

These records are generated from the config file. See the README for more info atm

"""

import tensorflow as tf
import numpy as np
import config
import cv2
import os

num_classes = config.num_classes

# helper function
def _bytes_feature(value):
   return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


"""

Main

"""
def setup(data_dir, dataset):

   # define the shape each image will be resized to
   SHAPE = (128, 128)

   # create a records directory if there isn't one already
   try:
      os.mkdir(data_dir+"/"+dataset+"/records")
   except:
      pass

   # define output filenames for the records - good to put them in the $DATA_DIR
   train_record = data_dir+"/"+dataset+"/records/train.tfrecord"
   test_record  = data_dir+"/"+dataset+"/records/test.tfrecord"
   #val_record   = data_dir+"/"+dataset+"/records/val.tfrecord"

   # check if they have run create_test_train_val first
   if os.path.isdir(data_dir+"/"+dataset+"/records/") is False:
      print "No records directory found in " + data_dir+"/"+dataset + " ... creating one"
      try:
         os.mkdir(data_dir+"/"+dataset+"/records")
      except:
         print "Could not create directory. Please run command: mkdir " + data_dir+"/"+dataset+"/records"
         print "Or check config.py for correct paths (permission error??)"
         exit()

   # tf writer for train test and val
   train_writer = tf.python_io.TFRecordWriter(train_record)
   test_writer  = tf.python_io.TFRecordWriter(test_record)
   #val_writer   = tf.python_io.TFRecordWriter(val_record)

   t = ["/test/", "/train/"]

   print "Creating records...\n"
   for a in t:
      for root, dirs, files in os.walk(data_dir+"/"+dataset+a):
         for d in dirs:
            label = d
            for r, l, images in os.walk(data_dir+"/"+dataset+a+label):
               for image in images:
                  # location of image
                  img_src = data_dir+"/"+dataset+a+label+"/"+image

                  # read image in
                  img = cv2.imread(img_src)

                  # resize image
                  img = cv2.resize(img, SHAPE, interpolation = cv2.INTER_CUBIC)

                  # flatten image
                  #img_flat = img.flatten()
                  img_flat = np.reshape(img,[1,SHAPE[0]*SHAPE[1]*3])
                  # _bytes_feature requires inputs as strings
                  example = tf.train.Example(features=tf.train.Features(feature={
                          'image': _bytes_feature(img_flat.tostring())}))
                  if a == "/test/":
                     test_writer.write(example.SerializeToString())
                  elif a == "/train/":
                     train_writer.write(example.SerializeToString())

if __name__ == "__main__":
   data_dir = config.data_dir
   dataset  = config.dataset
   setup(data_dir, dataset)
   print "Done"


