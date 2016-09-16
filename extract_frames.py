import cv2
import sys

vidcap = cv2.VideoCapture(sys.argv[1])
success,image = vidcap.read()
count = 0
success = True

# should probably give each show a show id
show_id = 1

# OR DRAW SPONGEBOB IN THE ANIMATION STYLE OF ARCHER OR ADVENTURE TIME
# so like train the shit out of archer, then pass in spongebob

# only save every two seconds
while success:
   success, image = vidcap.read()
   if count %60 == 0:
   	cv2.imwrite("images/"+str(show_id)+"_frame_%d.png" % count, image)
   	print "Saving frame " + str(count)
   	count += 1
   count += 1
