from PIL import Image
import sys
import cv2

pixelSizes = [3,4]
#colors = [10,15,20,25,30]

colors = [100]

for pixelSize in pixelSizes:
   for color in colors:
      image = Image.open(sys.argv[1])
      image.save('../images/output-original.png')
      image = image.resize((image.size[0]/pixelSize, image.size[1]/pixelSize), Image.NEAREST)
      image = image.resize((image.size[0]*pixelSize, image.size[1]*pixelSize), Image.NEAREST)
      image = image.convert('RGB').convert('P', palette=Image.ADAPTIVE, colors=color)
      pixel = image.load()
      image.save('../images/output-00'+str(pixelSize)+'-00'+str(color)+'.png')


