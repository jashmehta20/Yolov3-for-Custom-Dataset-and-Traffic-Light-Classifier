# Yolov3 Implementation for Custom and Pretrained dataset 
 
## Traffic Light Classfier
Deep learning network architecture called YOLOv3 was used for real time traffic light detection. It is a feature learning-based network that adopts 106 convolutional layers as its most powerful tool. Model weights and config file for the model was pre-trained. Pre-trained model can detect up to 80 different objects (e.g. Traffic signal, car, chair, etc.). I modified the model to just detect traffic signals from 80 different classes. A threshold for bounding box confidence was set as well as a threshold for Non-Maximal Supression. Binary mask images are generated for each colour of the signal (red, yellow and green). For each of the detected signals in each binary mask images, we compute the total number of white pixels which serve as an important variable in our decision rule.

The the colour with maximum white pixels classify the output as STOP, SLOW DOWN & GO.

## Instructions to run the code

- Gitclone the repository.
- Add path of new video to be tested or use live camera feed in the code **yolo.py**
- Run the **yolo.py** script to see the traffic light classification. 


Output:

![ezgif com-gif-maker-2](https://user-images.githubusercontent.com/81267080/139096677-81245125-e08a-470f-93a3-044567077365.gif)
