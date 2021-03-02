import cv2
import os

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_extractor(imagePath):
    img = cv2.imread(imagePath)
    cropped_face=None
    faces = face_classifier.detectMultiScale(img, 1.3, 5)
    if faces is None:
        cropped_face =  None
    else:
        for (x,y,w,h) in faces:
            x = x-10
            y = y-10
            cropped_face = img[y:y+h+20, x:x+w+20]
    if cropped_face is not None:
        face = cv2.resize(cropped_face, (400,400))
        cv2.imwrite(imagePath, face)
        return face
    else:
        return  None

