import tensorflow as tf
import numpy as np
import os
import sys
import numpy as np
import cv2
from optparse import OptionParser

sys.path.insert(0, '../input/')
sys.path.insert(0, '../model/')

import input_
import architecture
import time

def train(checkpoint_dir, record_file, batch_size):
   with tf.Graph().as_default():

      batch_size = int(batch_size)
      global_step = tf.Variable(0, name='global_step', trainable=False)

      train_size = 148623

      gameboy_images, hd_images = input_.inputs(record_file, batch_size, "train")

      logits = architecture.inference(batch_size, gameboy_images, "train")

      # loss is the l2 norm of my input vector (the image) and the output vector (generated image)
      loss = architecture.loss(hd_images, logits)
      
      train_op = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss, global_step=global_step)

      variables = tf.all_variables()
      init      = tf.initialize_all_variables()
      sess      = tf.Session()

      # image summary for tensorboard
      image_summary_t = tf.image_summary('images', gameboy_images, max_images=100)

      # summary for tensorboard graph
      summary_op = tf.merge_all_summaries()
    
      try:
         os.mkdir(checkpoint_dir)
      except:
         pass

      sess.run(init)
      print "\nRunning session\n"

      graph_def = sess.graph.as_graph_def(add_shapes=True)
      summary_writer = tf.train.SummaryWriter(checkpoint_dir+"training", graph_def=graph_def)

      # saver for the model
      saver = tf.train.Saver(tf.all_variables())
      
      tf.train.start_queue_runners(sess=sess)

      # restore previous model if one
      ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
      if ckpt and ckpt.model_checkpoint_path:
         print "Restoring previous model..."
         try:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print "Model restored"
         except:
            print "Could not restore model"
            pass

      step = int(sess.run(global_step))
      epoch_num = step/(train_size/batch_size)

      while True:
         _, loss_value, generated_image, imgs = sess.run([train_op, loss, logits, hd_images])
         print "Epoch: " + str(epoch_num) + " Step: " + str(sess.run(global_step)) + " Loss: " + str(loss_value)
         step += 1
         
         # save for tensorboard
         if step%(train_size/batch_size) == 0:
            summary_str = sess.run(summary_op)
            summary_writer.add_summary(summary_str, step)

            print "Finished epoech " + str(epoch_num) + " ....saving model"
            print
            saver.save(sess, checkpoint_dir+"epoch-"+str(epoch_num), global_step=global_step)
            print

def main(argv=None):
   parser = OptionParser(usage="usage")
   parser.add_option("-c", "--checkpoint_dir",          type="str")
   parser.add_option("-r", "--record_file",             type="str")
   parser.add_option("-b", "--batch_size", default=100, type="int")

   opts, args = parser.parse_args()
   opts = vars(opts)

   checkpoint_dir = opts['checkpoint_dir']
   batch_size     = opts['batch_size']
   record_file    = opts['record_file']

   if not os.path.isfile(record_file):
      print "Record file not found"
      exit()

   if checkpoint_dir is None:
      print "checkpoint_dir is required"
      exit()

   print
   print "checkpoint_dir: " + str(checkpoint_dir)
   print "record_file:    " + str(record_file)
   print "batch_size:     " + str(batch_size)
   print

   answer = raw_input("All correct?\n:")
   if answer == "n":
      exit()

   train(checkpoint_dir, record_file, batch_size)


if __name__ == "__main__":

   if sys.argv[1] == "--help" or sys.argv[1] == "-h" or len(sys.argv) < 2:
      print
      print "-c --checkpoint_dir <str> [path to save the model]"
      print "-r --record_file    <str> [path to the record file]"
      print "-b --batch_size     <int> [batch size]"
      print
      exit()


   tf.app.run()

