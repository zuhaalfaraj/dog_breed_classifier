# Dog Breed Classification using Pytorch

## Overview
This project aims to calssify dogs images to their correct class, and recognize human faces if the input isn't a dog image. So it has three cases: 
- if the image is for dog--> recognizes that, and returns its brees
- if the image is for human--> detects human faces.
- Gives an error if the input neither dog nor human

### Prerequisites
- Pytorch
- OpenCV
- PIL
- Numpy
- Matplotlib
- glob

### Data
- Dog Images: https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip
- Human Images: https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip

### Included materials
- dog_app.ipynb: this file contains the main class of the project that applies all functionalities 
- Scratch_Modoule.py: this file contains the builded from scratch CNN model
- haarcascades: this folder is required for a face detection opencv's library


### Results
The models are trained under computional power and time limitions.
- The accuracy of the builded cnn model is 24%
- The accuracy of the VGG16 model is a way better: 75%
