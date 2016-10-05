from PIL import Image
import sys
import cv2
import numpy as np

#color = 50
pixelSize = 12
SHAPE=(160,144)

# pixelate image
image = Image.open(sys.argv[1])
image = image.resize((image.size[0]/pixelSize, image.size[1]/pixelSize), Image.NEAREST)
image = image.resize((image.size[0]*pixelSize, image.size[1]*pixelSize), Image.NEAREST)

# convert to opencv
img = np.asarray(image)
img = img[:,:,::-1].copy()

# resize to gameboy color resolution
img = cv2.resize(img, SHAPE, interpolation = cv2.INTER_CUBIC)

# convert BACK to PIL so I can apply a color transformation
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = Image.fromarray(img)

img = img.convert(mode='P', colors=16)

# convert BACK to opencv
img = np.asarray(img)
img = img[:,:,::-1].copy()

#cv2.imwrite('resize.png', img)

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

