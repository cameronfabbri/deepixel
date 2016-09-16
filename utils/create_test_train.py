"""

Cameron Fabbri
06/08/2016

Script that will create symbolic links to different files for train and test.
The following folders are all contained in $DATA_DIR
A real directory structure is shown in tree.txt
tensorflow requires the folder structure to be:

   $TRAIN_DIR/dog/image0.jpeg
   $TRAIN_DIR/dog/image1.jpg
   $TRAIN_DIR/dog/image2.png
   ...
   $TRAIN_DIR/cat/weird-image.jpeg
   $TRAIN_DIR/cat/my-image.jpeg
   $TRAIN_DIR/cat/my-image.JPG
   ...
   
   $TEST_DIR/dog/imageA.jpeg
   $TEST_DIR/dog/imageB.jpg
   $TEST_DIR/dog/imageC.png
   ...
   $TEST_DIR/cat/weird-image.PNG

   img_data_dir/
      - images
      - train
      - test

"""

import ntpath
import os
import sys
from random import shuffle
import math
import glob
import math
import config

if __name__ == "__main__":

   train_perc = config.train_perc
   test_perc  = config.test_perc

   dataset = config.dataset

   data_dir = config.data_dir

   dataset_dir = data_dir + "/" + dataset + "/images"

   train_dir = data_dir + "/" + dataset + "/train"
   test_dir  = data_dir + "/" + dataset + "/test"

   label_list = list()

   hot_labels = list()

   for root, dirs, files in os.walk(dataset_dir):
      for label in dirs:
         label_list.append(label)

   # creating the test train and val directories
   try:
      os.mkdir(data_dir + "/" + dataset + "/test")
      os.mkdir(data_dir + "/" + dataset + "/train")
   except:
      pass

   # the label_list was created from the directory, so just go through that
   for label in label_list:
      # this will give the full path for the image in the folder
      image_list = glob.glob(dataset_dir+"/"+label+"/*.*")

      # shuffle the list so we get a random subset for train test val
      shuffle(image_list)
      train_num = int(math.ceil(train_perc*len(image_list)))
      train_set = image_list[:train_num]

      test_num  = int(math.ceil(test_perc*len(image_list)))
      test_set  = image_list[train_num:]

      # create a directory for the current label in test, val, and train
      try:
         os.mkdir(data_dir + "/" + dataset + "/train/" + label)
         os.mkdir(data_dir + "/" + dataset + "/test/" + label)
      except:
         continue

      # now symlink the images in each list to their respective directory
      for image in test_set:
         im_name = ntpath.basename(image)
         try:
            os.symlink(image, data_dir+"/"+dataset+"/test/"+label+"/"+im_name)
         except:
            continue

      for image in train_set:
         im_name = ntpath.basename(image)
         try:
            os.symlink(image, data_dir+"/"+dataset+"/train/"+label+"/"+im_name)
         except:
            continue

