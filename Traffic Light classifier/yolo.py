import cv2
import numpy as np

cap=cv2.VideoCapture('/Users/jashmehta/Downloads/IMG_6743.mp4')
whT=320
classesFiles='coco.names'
classNames=[]

confThreshold=0.5
nms_threshold=0.3
L_red1= np.array([0,100,100])
u_red1=np.array([10,255,255])
L_red2=np.array([160,100,100])
u_red2=np.array([180,255,255])
l_green=np.array([50,100,100])
u_green=np.array([90,255,255])
l_yellow=np.array([15,150,150])
u_yellow=np.array([35,255,255])

with open(classesFiles,'rt') as f:
    classNames=f.read().rstrip('\n').split('\n')

modelConfiguration='/Users/jashmehta/Desktop/cvproject/Yolov3-for-Custom-Dataset-and-Traffic-Light-Classifier/Traffic Light classifier/yolov3-320.cfg'
modelWeights='/Users/jashmehta/Desktop/cvproject/Yolov3-for-Custom-Dataset-and-Traffic-Light-Classifier/Traffic Light classifier/yolov3.weights'

net=cv2.dnn.readNetFromDarknet(modelConfiguration,modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
layerNames=net.getLayerNames()
outputNames = [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
def Status(X1):
   
    ''' This funtion declare if to Go, Slow Down and Stop '''
    
    Hsv= cv2.cvtColor(X1, cv2.COLOR_BGR2HSV)
    hsv.append(Hsv)
    countg=0
    countr=0
    county=0
    
    MASK1= cv2.inRange(Hsv,L_red1,u_red1)
    mask1.append(MASK1)
    
    MASK2= cv2.inRange(Hsv,L_red2,u_red2)
    mask2.append(MASK2)
    MASKG= cv2.inRange(Hsv, l_green,u_green)
    maskg.append(MASKG)
    
    MASKy= cv2.inRange(Hsv, l_yellow, u_yellow)
    masky.append(MASKy)
    MASKr= cv2.add(MASK1, MASK2)
    maskr.append(MASKr)
    dime1=np.shape(MASKr)
    dime.append(dime1)
    (H1,W1)= dime1
    for j in range(H1):
        for k in range(W1):
           if MASKr[j][k]==255:
               countr+=1
           if MASKG[j][k]==255:
               countg+=1
           if MASKy[j][k]==255:
               county+=1
    
    L= [0, countr, county, countg]
    final_color= L.index(max(L))
    
    if final_color==1:
        status="STOP"
    elif final_color==2:
        status="SLOW DOWN"
    elif final_color==3:
        status="GO"
    else:
        status=""
    return status


def findObjects(outputs,img):
    hT,wT,cT=img.shape
    bbox=[]
    classIds=[]
    confs=[]
    X1=[]
    hsv=[]
    mask1=[]
    mask2=[]
    maskr=[]
    masky=[]
    maskg=[]
    dime=[]

    for output in outputs:
        for det in output:
            scores=det[5:]
            classId=np.argmax(scores)
            confidence=scores[classId]
            if confidence>confThreshold and classId==9:
                w,h=int(det[2]*wT),int(det[3]*hT)
                x,y=int((det[0]*wT) - w/2),int((det[1]*hT)-h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))
    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nms_threshold)
    frame_new=img.copy()

    for i in indices:
        i=i[0]
        box=bbox[i]
        x,y,w,h=box[0],box[1],box[2],box[3]
        X.append(x)
        Y.append(y)
        X1.append(frame_new[Y[i]:Y[i] + h, X[i]:X[i] + w])
        status= Status(X1[i])
        cv2.rectangle(img, (x,y),(x+w,y+h) , (255,0,255),2)
        cv2.putText(img,status,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,255),2)

while True:
    success,img=cap.read()
    blob=cv2.dnn.blobFromImage(img,1/255,(whT,whT),[0,0,0],1,crop=False)
    net.setInput(blob)

    outputs=net.forward(outputNames)

    findObjects(outputs, img)
    cv2.imshow("image", img)

    key=cv2.waitKey(1)
    if key == 27:
        break