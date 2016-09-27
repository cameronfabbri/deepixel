import cv2

f = 'image_303.bmp'

img = cv2.imread(f)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
SHAPE = (224,224)
img = cv2.resize(img, SHAPE, interpolation=cv2.INTER_CUBIC)

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

