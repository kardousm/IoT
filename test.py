from imutils.video import VideoStream
import cv2
import imutils
import numpy
import pyrebase

config = {
    "apiKey": "AIzaSyAlup_ohoCJ9FZ_2z-E3YUXcfe_Rwbdw0E",
    "authDomain": "maneesh-iot.firebaseapp.com",
    "databaseURL": "https://maneesh-iot.firebaseio.com",
    "projectId": "maneesh-iot",
    "storageBucket": "maneesh-iot.appspot.com",
    "messagingSenderId": "100571796856",
    "appId": "1:100571796856:ios:5a52d4ad6e0f5df844248d",
    "serviceAccount": "maneesh-iot-firebase-adminsdk-m9gqi-c10e487c04.json"
}

firebase = pyrebase.initialize_app(config)
storage = firebase.storage()


url = storage.child("Ned Stark/Ned Stark.jpg").get_url(None)

cv = cv2.imread('test/Ned Stark.jpg')
im = imutils.url_to_image(url)
print(numpy.array_equal(im, cv))