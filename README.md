# Yolov3 Implementation for Custom and Pretrained dataset 
 
## Traffic Light Classfier
Deep learning network architecture called YOLOv3 was used for real time object detection. It is a feature learning-based network that adopts 106 convolutional layers as its most powerful tool. Model weights and config file for the model was pre-trained. Pre-trained model can detect up to 80 different objects (e.g. Traffic signal, car, chair, etc.). We modified the model to just detect traffic signal from 80 different classes. We set a threshold where we specify that bounding box with probability or confidence less than 75% is discarded. Binary mask images are generated for each colour of the signal (red, yellow and green). For each of the detected signals in each binary mask images, we compute the total number of white pixels which serve as an important variable in our decision rule.

The the colour with maximum white pixels classify the output as STOP, SLOW DOWN & GO.
