# create_data.py
# script used to create a dataset of someone's face

# import openCV, numpy, and system-related modules
import cv2, numpy, sys, os, time
# change the paths below to the location where these files are on your machine
haar_file = '/home/USER/Workspaces/Python/openCV/Facial-recognition/haarcascade_frontalface_default.xml'
# All of the faces data (images) will be stored here
datasets = '/home/USER/Workspaces/Python/openCV/Facial-recognition/faces'
# Sub dataset in 'faces' folder. Each folder is specific to an individual person
# change the name below when creating a new dataset for a new person
sub_dataset = 'Abdullah'

path = os.path.join(datasets, sub_dataset)
if not os.path.isdir(path):
    os.mkdir(path)

# defining the size of images
(width, height) = (130, 100)

face_cascade = cv2.CascadeClassifier(haar_file)
# use '0' for internal (built-in) webcam or '1' for external ones
webcam = cv2.VideoCapture(0)
print("Webcam is open? ", webcam.isOpened())
time.sleep(1)
#Takes pictures of detected face and saves them
count = 1
print("Taking pictures...")
# this takes 30 pictures of your face. Change this number if you want the classifier to be more accurate.
# Having too many images, however, might slow down the program and use more of your CPU (to train the classifier)
while count < 31:
    ret_val, im = webcam.read()
    if ret_val == True:
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 4)
        for (x,y,w,h) in faces:
            cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
            face = gray[y:y + h, x:x + w]
            # resize the face images for consistency
            face_resize = cv2.resize(face, (width, height))
            # save images with their corresponding number
            cv2.imwrite('%s/%s.png' % (path,count), face_resize)
        count += 1
        # display the openCV window
        cv2.imshow('OpenCV', im)
        key = cv2.waitKey(20)
        if key == 27:
            break
print("Sub dataset for your face has been created.")
webcam.release()
cv2.destroyAllWindows()