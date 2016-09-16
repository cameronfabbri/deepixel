"""
Cameron Fabbri
9/16/2016

Extracting frames from tv shows, simply give a folder path to the tv show

"""

import cv2
import sys
import glob
import os

"""
assumes the directory structure:
	tv show/
 		- tv show season 1/
 			- episode1.mkv
 			- episode2.mkv
 			...
 		...
"""

tv_show_path    = sys.argv[1]
download_folder = sys.argv[2]

for root, dirs, files in os.walk(tv_show_path):
    for season in dirs:
        for r, d, files in os.walk(root+season):
            for episode in files:
                episode_frame_count = 0
                episode_name = episode.split(".m")[0]
                episode = root+season+"/"+episode
                vidcap = cv2.VideoCapture(episode)
                success = True
                try:
                    os.mkdir(download_folder+episode_name)
                except:
                    pass
                while success:
                    success, image = vidcap.read()
                    if episode_frame_count%60 == 0:
                        print "Saving frame " + str(episode_frame_count)
                        cv2.imwrite(download_folder+episode_name+"/"+str(episode_frame_count)+".jpg", image)
                        episode_frame_count += 1
                    else:
                        episode_frame_count += 1
                print "Done with episode " + str(episode_name)
        print "Done with season " + str(season)
        exit()


            

# OR DRAW SPONGEBOB IN THE ANIMATION STYLE OF ARCHER OR ADVENTURE TIME
# so like train the shit out of archer, then pass in spongebob

# only save every two seconds
"""
while success:
   success, image = vidcap.read()
   if count %60 == 0:
   	cv2.imwrite("images/"+str(show_id)+"_frame_%d.png" % count, image)
   	print "Saving frame " + str(count)
   	count += 1
   count += 1
"""
