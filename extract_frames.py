import cv2
import sys

vidcap = cv2.VideoCapture(sys.argv[1])
success,image = vidcap.read()
count = 0
success = True

# only save every 
while success:
   success,image = vidcap.read()
   cv2.imwrite("temp/frame%d.png" % count, image)     # save frame as JPEG file
   count += 1
   print count

