import cv2
img = cv2.imread('imges/street.jpg')




classname=[]
classfiles='files/thing.names'
with open(classfiles,'rt') as d:
    classname=d.read().rstrip('\n').split('\n')

print(classname)

z='files/frozen_inference_graph.pb'
y='files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'

net = cv2.dnn_DetectionModel(z,y) #for examination
net.setInputSize(320,230) # width and height
net.setInputScale(1.0/127.5) #  measurement
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)# calor system

classIds,confs,bbox= net.detect(img,confThreshold=0.5)  
#print(classIds,bbox)
for classId,confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
    cv2.rectangle(img,box,color=(0,255,0),thickness=3)
    cv2.putText(img,classname[classId-1],
                (box[0]+10,box[1]+20),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),thickness=2)

cv2.imshow('program',img)
cv2.waitKey(0)

