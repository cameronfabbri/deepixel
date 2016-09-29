from PIL import Image
import sys

backgroundColor = (0,)*3
pixelSizes = range(1,20)

for pixelSize in pixelSizes:
   image = Image.open(sys.argv[1])
   image = image.resize((image.size[0]/pixelSize, image.size[1]/pixelSize), Image.NEAREST)
   image = image.resize((image.size[0]*pixelSize, image.size[1]*pixelSize), Image.NEAREST)
   pixel = image.load()
   image.save('output-'+str(pixelSize)+'.png')
