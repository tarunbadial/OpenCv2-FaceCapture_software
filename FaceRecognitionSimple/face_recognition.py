import cv2
import numpy as np
import os 

#path_find
def path_checking(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)
path_checking("saved_model/")
def getname(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)] 
    name={}
    for imagePath in imagePaths:
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        nam = os.path.split(imagePath)[-1].split(".")[0]
        name[id]=nam
    return name
name = getname('training_cp')
print(name)
# Create face recognization
detect = cv2.cv2.face.LBPHFaceRecognizer_create()
# load model
detect.read('saved_model/s_model.yml')
#  Frontal Face detection
faceCascade = cv2.CascadeClassifier("frontalface_default.xml");
# font style
font = cv2.FONT_HERSHEY_SIMPLEX
cam = cv2.VideoCapture(0)

while True:
    ret,img =cam.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.2,5) 
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x-20,y-20), (x+w+20,y+h+20), (0,255,0), 2)
        Id, confidence = detect.predict(gray[y:y+h,x:x+w]) 
        print(Id )
        print(confidence)
        if Id == Id:
            Id = "%s %d"%(name[Id],Id)  
        # Set rectangle around face and name of the person
        cv2.rectangle(img, (x-22,y-40), (x+w+22, y-10), (0,255,0), -1)
        cv2.putText(img, str(Id), (x,y-20), font, 0.8, (255,255,255), 2)
    cv2.imshow('Live_detect',img) 
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
        
    
cam.release()
# Close all windows
cv2.destroyAllWindows()
