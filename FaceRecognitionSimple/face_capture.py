#Important Libararies
from tkinter import *
import os
import cv2
from tkinter import messagebox as ms
from tkinter import ttk	

class capture_face:
    def __init__(self,master):
        self.master=master
        self.notes=Label(master,text="Please Insert the Unique User Id !!",font=("bold",12),fg="red")
        self.user_ilabel=Label(master,text="User Id: ",font=("bold",10))
        self.user_id=Entry(master)
        self.user_label=Label(master,text="User Name: ",font=("bold",10))
        self.user_name=Entry(master)
        self.capture=Button(master,text ="Capture",fg='black',width=4,height=1,command=self.shot) 
        self.clear=Button(master,text ="Clear",fg='black',width=2,height=1,command=self.clear_text)
        self.submit=Button(master,text ="Submit",fg='black',width=3,height=1,command=self.submit)
      
    #locations set of labels and buttons
        self.notes.place(x=10,y=20)
        self.user_ilabel.place(x=10,y=50)
        self.user_id.place(x=100,y=50)
        self.user_label.place(x=10,y=80)
        self.user_name.place(x=100,y=80)
        self.capture.place(x=20,y=120)
        self.clear.place(x=120,y=120)
        self.submit.place(x=200,y=120)
    
	# Path Assure Directories
    def assure_path_exists(path):
        dir = os.path.dirname(path)
        if not os.path.exists(dir):
            os.makedirs(dir)
    assure_path_exists("users/")
    
	# Image Shots here
    def shot(self):
        try:
            vid_cam = cv2.VideoCapture(0)
            face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            count = 0
            img_count=102
            while(True):
                _, image_frame = vid_cam.read()
                gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)
                faces = face_detector.detectMultiScale(gray, 1.3, 5)
                for (x,y,w,h) in faces:
                    cv2.rectangle(image_frame, (x,y), (x+w,y+h), (255,0,0), 2)
                    img_count += 1
                    cv2.imwrite("users/"+self.user_name.get()+"." + str(self.user_id.get()) + '.' + str(img_count) + ".jpg", gray[y:y+h,x:x+w])
                    cv2.imshow('frame', image_frame)
                    count+=1
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    ms.showinfo("Greating!!","User Added Sucessfull")
                    break
                elif count>100:
                    ms.showinfo("Greating!!","User Added Sucessfull")
                    break
            vid_cam.release()
            cv2.destroyAllWindows()
        except:   
            ms.showinfo("Aborted!!","User Not Added Sucessfull")
     
	 # Reset Text Feilds  
    def clear_text(self):
        self.user_name.delete(0, END)
        self.user_id.delete(0, END)
    #Submitted
    def submit(self):
        def assure_training_exists(path):
            dir = os.path.dirname(path)
            if not os.path.exists(dir):
               os.makedirs(dir)
        assure_training_exists("training_cp/")
        os.system("cp -rv users/* training_cp/")
        os.system("pwd")
        os.system("python3 training.py")
        os.system("rm  -rvf users/*")
        ms.showinfo("Sucessfull!!","User Data Submit Sucessfull")

#GUI Container Create            
root=Tk();
ob=capture_face(root);
root.title("Face Capture");
#root.iconbitmap('Capture.ico')
root.geometry("300x200+150+150")
root.mainloop()
