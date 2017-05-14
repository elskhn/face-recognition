# face_recognize.py

size = 4
import cv2, sys, numpy, os
# haar_file path
haar_file = '/home/USER/Workspaces/Python/openCV/Facial-recognition/haarcascade_frontalface_default.xml'
# main datasets folder path
datasets = '/home/USER/Workspaces/Python/openCV/Facial-recognition/faces'

print('Training classifier...')
# Create a list of images and a list of corresponding names along with a unique id
(images, labels, names, id) = ([], [], {}, 0)
for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(datasets, subdir)
        for filename in os.listdir(subjectpath):
            path = subjectpath + '/' + filename
            label = id
            images.append(cv2.imread(path, 0))
            labels.append(int(label))
        id += 1
(width, height) = (130, 100)

# Create a numpy array from the lists above
(images, labels) = [numpy.array(lists) for lists in [images, labels]]

# OpenCV trains a model from the images using the FisherFace algorithm
model = cv2.createFisherFaceRecognizer()
model.train(images, labels)
print('Classifier trained!')
# use fisherRecognizer on webcam stream
face_cascade = cv2.CascadeClassifier(haar_file)
webcam = cv2.VideoCapture(0)
print('Attempting to recognize faces...')
while True:
    (_, im) = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # detect faces using haar_cacade and draw a rectangle around them
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        # colour = bgr format
        cv2.rectangle(im,(x,y),(x + w,y + h),(0, 255, 255),2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))
        # try to recognize the face(s)
        prediction = model.predict(face_resize)
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 255), 2)
        # if face is recognized, display the corresponding name
        if prediction[1]<500:
	       cv2.putText(im,'%s' % (names[prediction[0]]),(x + 10, (y + 22) + h), cv2.FONT_HERSHEY_PLAIN,1.5,(20,205,20), 2)
        # if face is unknown (if classifier is not trained on this face), show 'Unknown' text...
    	else:
    	  cv2.putText(im,'Unknown',(x + 10, (y + 22) + h), cv2.FONT_HERSHEY_PLAIN,1.5,(65,65, 255), 2)

    # show window and set the window title
    cv2.imshow('OpenCV Face Recognition -  esc to close', im)
    key = cv2.waitKey(10)
    # esc to quit applet
    if key == 27:
        break