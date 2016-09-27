from PIL import Image
import sys

backgroundColor = (0,)*3
pixelSize = 7

image = Image.open(sys.argv[1])
#image.show()
image = image.resize((image.size[0]/pixelSize, image.size[1]/pixelSize), Image.NEAREST)
image = image.resize((image.size[0]*pixelSize, image.size[1]*pixelSize), Image.NEAREST)
pixel = image.load()

'''
for i in range(0,image.size[0],pixelSize):
  for j in range(0,image.size[1],pixelSize):
    for r in range(pixelSize):
      pixel[i+r,j] = backgroundColor
      pixel[i,j+r] = backgroundColor
'''


image.show()

#image.save('output.png')
