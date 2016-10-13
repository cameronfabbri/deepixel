# deepixel emulator

This is a deep learning approach to depixelizing images, with a goal to fully integrate
it into an emulator to provide real-time HD gameplay.

## Approach
The idea is to resize high definition images from side-scrolling video games to the original
Gameboy Color resolution (160,144) to use as a training set. The resized HD image will serve as the input to the
network, and the output will be compared with the original unsized image.

### Original image taken from Donkey Kong Country Tropical Freeze Wii U
![original](https://github.com/cameronfabbri/deepixel/blob/master/images/output-original.png?raw=true)

### Resized Donkey Kong Country Tropical Freeze 
![gbco](https://raw.githubusercontent.com/cameronfabbri/deepixel/master/images/oo_.png)

### Original image taken from Donkey Kong Country Gameboy Color
![gbc](https://raw.githubusercontent.com/cameronfabbri/deepixel/master/images/dk2.png)


