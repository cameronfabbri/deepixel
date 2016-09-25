#!/bin/bash

for file in "$1"/*.*; do
   destination="${file%.*}"
   echo "Extracting from $file..."
   mkdir -p "$destination"
   ./ffmpeg_g.exe -i "$file" -r 1/1 "$destination/image_%03d.bmp"
done
