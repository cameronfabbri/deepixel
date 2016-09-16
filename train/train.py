import tensorflow as tf
import numpy as np
import os
import sys
import numpy as np
import cv2

sys.path.insert(0, '../utils/')
sys.path.insert(0, '../inputs/')
sys.path.insert(0, '../model/')

import config
import input_
import architecture
import time

checkpoint_dir = config.checkpoint_dir
batch_size = config.batch_size


def train():
   with tf.Graph().as_default():
      global_step = tf.Variable(0, trainable=False)

      images = input_.inputs("train", batch_size)
      #images = tf.div(images, 255)

      logits = architecture.inference(images, "train")

      # loss is the l2 norm of my input vector (the image) and the output vector (generated image)
      loss = architecture.loss(images, logits)
      
      train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)

      variables = tf.all_variables()

      init = tf.initialize_all_variables()

      sess = tf.Session()

      # summary for tensorboard graph
      summary_op = tf.merge_all_summaries()
      
      sess.run(init)
      print "\nRunning session\n"

      graph_def = sess.graph.as_graph_def(add_shapes=True)
      summary_writer = tf.train.SummaryWriter(checkpoint_dir+"training", graph_def=graph_def)

      # saver for the model
      saver = tf.train.Saver(tf.all_variables())
      
      tf.train.start_queue_runners(sess=sess)

      #for step in xrange(10000):
      step = 0
      while True:
         step += 1
         _, loss_value, generated_image, imgs = sess.run([train_op, loss, logits, images])
         #print logs
         print "Step: " + str(step) + " Loss: " + str(loss_value)

         # save for tensorboard
         if step%5000 == 0 and step != 0:
            summary_str = sess.run(summary_op)
            summary_writer.add_summary(summary_str, step)

            print "Saving model..."
            saver.save(sess, checkpoint_dir+"training", global_step=step)
            """
            c = 1
            for im, gen in zip(imgs, generated_image):
               im = np.uint8(im)
               gen = np.uint8(gen)
               cv2.imshow('img', im)
               cv2.imshow('gen', gen)
               #cv2.waitKey(0)
               time.sleep(5)
               cv2.destroyAllWindows()
               if c == 3:
                  break
               c += 1
            """
            

def main(argv=None):
   train()


if __name__ == "__main__":
   tf.app.run()

