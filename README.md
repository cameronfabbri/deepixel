# deepixel emulator

This is a deep learning approach to depixelizing images, with a goal to fully integrate
it into an emulator to provide real-time HD gameplay.

## Approach
The idea is to generate pixelated versions of high definition images from side-scrolling video
games and cartoons to use as a training set. The pixelated image will serve as the input to the
network, and the output will be compared with the original unpixelated image. The hope is that 
the pixelated images will be representative of the retro versions of Gameboy Color games, so that 
after the network learns to depixelate the image, it can be applied to older video games.

Seen from left to right:
Original image, pixelated image, original Gameboy Color image

![origanl](https://github.com/cameronfabbri/deepixel/blob/master/images/output-original.png?raw=true)![pixelated](https://github.com/cameronfabbri/deepixel/blob/master/images/output-6.png?raw=true)![gbc](http://199.101.98.242/media/images/33501-Donkey_Kong_Country_(Europe)_(En,Fr,De)-3.jpg)

