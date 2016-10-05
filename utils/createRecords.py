from PIL import Image
import tensorflow as tf
from tqdm import tqdm
import sys
import cv2
import numpy as np
import os
import fnmatch

pixelSizes = [4,5]
colors = [50, 100, 150, 200]

def setup(folder, dest_dir):
   
   folder = "/home/neptune/data_dir/games/"
   pattern = "*.png"
   fileList = list()

   for d, s, fList in os.walk(folder):
      for filename in fList:
         if fnmatch.fnmatch(filename, pattern):
            fileList.append(os.path.join(d,filename))

   for image_name in fileList:
      for pixelSize in pixelSizes:
         for color in colors:
            image = Image.open(image_name)
            image = image.resize((image.size[0]/pixelSize, image.size[1]/pixelSize), Image.NEAREST)
            image = image.resize((image.size[0]*pixelSize, image.size[1]*pixelSize), Image.NEAREST)
            image = image.convert('RGB').convert('P', palette=Image.ADAPTIVE, colors=color)
            pixel = image.load()
            
            im = np.array(image)
      

if __name__ == "__main__":

   if len(sys.argv) < 3:
      print "Usage: python createRecords.py [source directory] [destination directory]"
      exit()

   folder = sys.argv[1]
   dest_dir  = sys.argv[2]

   setup(folder, dest_dir)

