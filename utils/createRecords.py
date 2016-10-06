from PIL import Image
import tensorflow as tf
from tqdm import tqdm
import sys
import cv2
import numpy as np
import os
import fnmatch

in_shape  = (160,144)
out_shape = (1280, 720)

# helper function
def _bytes_feature(value):
   return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def run(folder, dest_dir):
   
   record = dest_dir+"images.tfrecord"

   count = 0

   print record
   if os.path.isfile(record):
      if raw_input("Record file already exists! Overwrite? (y/n)") == "n":
         exit()

   record_writer = tf.python_io.TFRecordWriter(record)

   folder = "/home/neptune/data_dir/games/"
   pattern = "*.png"
   fileList = list()

   for d, s, fList in os.walk(folder):
      for filename in fList:
         if fnmatch.fnmatch(filename, pattern):
            fileList.append(os.path.join(d,filename))

   for image_name in tqdm(fileList):
      try:
      # read image
      try:
         img = cv2.imread(image_name)
      except:
         print "corrupt image: " + str(image_name)
         continue

      # make a copy of the image for the desired output of the network
      try:
         hd_img = img.copy()
      except:
         print "Couldn't copy image " + str(image_name)
         pass
      
      # resize to be 720p
      try:
         hd_img = cv2.resize(img, out_shape, interpolation=cv2.INTER_CUBIC)
         hd_height, hd_width, hd_channels = hd_img.shape

         # resize to gameboy color dimensions
         img = cv2.resize(img, in_shape, interpolation=cv2.INTER_CUBIC)
         height, width, channels = img.shape
      except:
         print "Error with " + str(image_name)
         continue

      # change to 15-bit colorspace
      for i in range(0, height):
         for j in range(0, width):
            r = img[i,j][0]
            g = img[i,j][1]
            b = img[i,j][2]
            r_p = (r*31/255)*(255/31)
            g_p = (g*31/255)*(255/31)
            b_p = (b*31/255)*(255/31)

            img[i,j] = [r_p, g_p, b_p]
            
      # flatten image
      img_flat = np.reshape(img, [1, in_shape[0]*in_shape[1]*3])

      # flatten hd image
      hd_img_flat = np.reshape(hd_img, [1, hd_height*hd_width*3])

      example = tf.train.Example(features=tf.train.Features(feature={
         'hd_image': _bytes_feature(hd_img_flat.tostring()),
         'img'     : _bytes_feature(img_flat.tostring())}))

      try:
         record_writer.write(example.SerializeToString())
      except:
         raise
      count += 1
   print "Created " + str(count) + " records"

if __name__ == "__main__":

   if len(sys.argv) < 3:
      print "Usage: python createRecords.py [source directory] [destination directory]"
      exit()

   folder = sys.argv[1]
   dest_dir  = sys.argv[2]

   run(folder, dest_dir)

