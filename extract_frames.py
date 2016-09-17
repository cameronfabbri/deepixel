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

show_name = tv_show_path.split("/")[-2]

for root, dirs, files in os.walk(tv_show_path):
    for season in dirs:
        if season == "Adventure Time Season 4":
            print "Skipping!!!"
            continue
        for r, d, files in os.walk(root+season):
            for episode in files:
                episode_frame_count = 0
                episode_name = episode.split(".m")[0]
                episode = root+season+"/"+episode
                vidcap = cv2.VideoCapture(episode)
                success = True
                try:
                    os.mkdir(download_folder+show_name)
                    os.mkdir(download_folder+show_name+"/"+season)
                    os.mkdir(download_folder+show_name+"/"+season+"/"+episode_name)
                except:
                    pass
                while success:
                    success, image = vidcap.read()
                    # extract every second of video
                    if episode_frame_count%30 == 0:
                        print "Saving frame " + str(episode_frame_count)
                        cv2.imwrite(download_folder+show_name+"/"+season+"/"+episode_name+"/"+str(episode_frame_count)+".png", image)
                        episode_frame_count += 1
                    else:
                        episode_frame_count += 1
                print
                print "Done with episode " + str(episode_name)
                print
        print
        print "Done with season " + str(season)
        print
print "Done with show " + str(show_name)
