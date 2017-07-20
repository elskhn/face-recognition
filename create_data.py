# OpenCV Facial Recognition
# Written + Documented by Abdullah Khan

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

# join the paths to include the sub_dataset folder
path = os.path.join(datasets, sub_dataset)
# if sub_dataset folder doesn't already exist, make the folder with the name defined above
if not os.path.isdir(path):
    os.mkdir(path)

# defining the size of images
(width, height) = (130, 100)

face_cascade = cv2.CascadeClassifier(haar_file)
# use '0' for internal (built-in) webcam or '1' for external ones
webcam = cv2.VideoCapture(0)
# returns true or false (if the camera is on or not)
print("Webcam is open? ", webcam.isOpened())
# wait for the camera to turn on (just to be safe, in case the camera needs time to load up)
time.sleep(2)
#Takes pictures of detected face and saves them
count = 1
print("Taking pictures...")
# this takes 100 pictures of your face. Change this number if you want.
# Having too many images, however, might slow down the program
while count < 101:
    # im = camera stream
    ret_val, im = webcam.read()
    # if it recieves something from the webcam...
    if ret_val == True:
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        # detect face using the haar cascade file
        faces = face_cascade.detectMultiScale(gray, 1.3, 4)
        for (x,y,w,h) in faces:
            # draws a rectangle around your face when taking pictures
            # this is to create a ROI (region of interest) so it only takes pictures of your face
            cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
            # define 'face' as the inside of the rectangle we made above and make it grayscale
            face = gray[y:y + h, x:x + w]
            # resize the face images to the size of the 'face' variable above (i.e: area captured inside of the rectangle)
            face_resize = cv2.resize(face, (width, height))
            # save images with their corresponding number
            cv2.imwrite('%s/%s.png' % (path,count), face_resize)
        count += 1
        # display the openCV window
        cv2.imshow('OpenCV', im)
        key = cv2.waitKey(20)
        # press esc to stop the loop
        if key == 27:
            break
print("Sub dataset for your face has been created.")
webcam.release()
cv2.destroyAllWindows()
