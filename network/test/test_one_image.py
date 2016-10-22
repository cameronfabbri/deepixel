import tensorflow as tf
import numpy as np
import os
import sys
import numpy as np
import cv2
from optparse import OptionParser

sys.path.insert(0, '../input/')
sys.path.insert(0, '../model/')

import architecture
import time

def test(checkpoint_dir, record_file, image_path):
   with tf.Graph().as_default():

      input_image = tf.placeholder(tf.float32, shape=(10,144,160,3))

      logits = architecture.inference(1, input_image, "train")

      # summary for tensorboard graph
      summary_op = tf.merge_all_summaries()

      variables = tf.all_variables()
      init      = tf.initialize_all_variables()
      sess      = tf.Session()

      # saver for the model
      saver = tf.train.Saver(tf.all_variables())
      
      tf.train.start_queue_runners(sess=sess)


      # restore previous model if one
      print checkpoint_dir
      ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
      if ckpt and ckpt.model_checkpoint_path:
         print "Restoring previous model..." + ckpt.model_checkpoint_path
         try:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print "Model restored"
         except:
            print "Could not restore model"
            pass


      # Summary op
      graph_def = sess.graph.as_graph_def(add_shapes=True)

     
      # run on image 
      img = cv2.imread(image_path)
      print(img)
      img = cv2.resize(img, (160,144)) / 255.0
      #img = np.transpose(img, (1,0,2))
      fake = np.zeros((10,144,160,3))
      fake[0,:,:,:] = img
      high_res = sess.run([logits],feed_dict={input_image:fake})[0]
      high_res = np.uint8(np.maximum(high_res,0)*255)
      cv2.imwrite('hd_mario.jpg', high_res[0,:,:,:])
      

def main(argv=None):
   parser = OptionParser(usage="usage")
   parser.add_option("-c", "--checkpoint_dir",          type="str")
   parser.add_option("-r", "--record_file",             type="str")
   parser.add_option("-i", "--image", default="mario_test.png", type="str")

   opts, args = parser.parse_args()
   opts = vars(opts)

   checkpoint_dir = opts['checkpoint_dir']
   record_file    = opts['record_file']
   image    = opts['image']

   if not os.path.isfile(record_file):
      print "Record file not found"
      exit()

   if checkpoint_dir is None:
      print "checkpoint_dir is required"
      exit()

   print
   print "checkpoint_dir: " + str(checkpoint_dir)
   print "record_file:    " + str(record_file)
   print "image:     " + str(image)
   print

   answer = raw_input("All correct?\n:")
   if answer == "n":
      exit()

   test(checkpoint_dir, record_file, image)


if __name__ == "__main__":

   if sys.argv[1] == "--help" or sys.argv[1] == "-h" or len(sys.argv) < 2:
      print
      print "-c --checkpoint_dir <str> [path to save the model]"
      print "-r --record_file    <str> [path to the record file]"
      print "-b --image          <str> [image to run on]"
      print
      exit()


   tf.app.run()

