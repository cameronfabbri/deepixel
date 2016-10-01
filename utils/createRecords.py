from PIL import Image
import sys
import cv2
import numpy as np
pixelSizes = [4,5]

colors = [50, 100, 150, 200]

for pixelSize in pixelSizes:
   for color in colors:
      image = Image.open(sys.argv[1])
      image.save('../images/output-original.png')
      image = image.resize((image.size[0]/pixelSize, image.size[1]/pixelSize), Image.NEAREST)
      image = image.resize((image.size[0]*pixelSize, image.size[1]*pixelSize), Image.NEAREST)
      image = image.convert('RGB').convert('P', palette=Image.ADAPTIVE, colors=color)
      pixel = image.load()
      
      im = np.array(image)
      image.save('../images/output-00'+str(pixelSize)+'-00'+str(color)+'.png')



