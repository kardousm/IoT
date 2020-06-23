# USAGE
# python recognize_video.py --detector face_detection_model \
#	--embedding-model openface_nn4.small2.v1.t7 \
#	--recognizer output/recognizer.pickle \
#	--le output/le.pickle

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os
import pyrebase

config = {
    "apiKey": "AIzaSyDVvv_NerGXIl7xXVDiZB6hWJoQAb0sZDQ",
    "authDomain": "test-2c93b.firebaseapp.com",
    "databaseURL": "https://test-2c93b.firebaseio.com",
    "projectId": "test-2c93b",
    "storageBucket": "test-2c93b.appspot.com",
    "messagingSenderId": "782137670239",
    "appId": "1:782137670239:web:4e9b9983730397f161f9ea",
    "measurementId": "G-W7PVLBLNV5",
    "serviceAccount": "test-2c93b-firebase-adminsdk-fbs0y-24184fce57.json"
}

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--detector", required=True,
                help="path to OpenCV's deep learning face detector")
ap.add_argument("-m", "--embedding-model", required=True,
                help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-r", "--recognizer", required=True,
                help="path to model trained to recognize faces")
ap.add_argument("-l", "--le", required=True,
                help="path to label encoder")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
                              "res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load our serialized face embedding model from disk
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open(args["recognizer"], "rb").read())
le = pickle.loads(open(args["le"], "rb").read())

# initialize the video stream, then allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
#vs = cv2.VideoCapture(0)

time.sleep(1.0)

# start the FPS throughput estimator
fps = FPS().start()
name_list = {}
temp1 = 0
temp2 = None
temp3 = False
data = {}




def ignore_first():
    global temp1
    global temp2
    if temp1 == 0:
        temp2 = False
        temp1 = 1
        return temp2
    else:
        return True


def stream_handler(post):
    global temp3
    if ignore_first():
        print('Database has been updated. Re-running ML Model.')
        os.system("python3 extract_embeddings.py --dataset test --embeddings output/embeddings.pickle --detector face_detection_model --embedding-model openface_nn4.small2.v1.t7")
        os.system("python3 train_model.py --embeddings output/embeddings.pickle --recognizer output/recognizer.pickle --le output/le.pickle")
        print("New facial recognition window is currently opening")
        temp3 = True
        os.system("python3 recognize_video.py --detector face_detection_model --embedding-model openface_nn4.small2.v1.t7 --recognizer output/recognizer.pickle --le output/le.pickle")




firebase = pyrebase.initialize_app(config)
db = firebase.database()

my_stream = db.child("Users").child("Profiles").stream(stream_handler, None)
data2 = {"Exit": "None"}
db.child('Users').child('Exit Notifications').set(data2)

#data1 = {"Mario": 10, "Maneesh":10, "Liz": 10, "Manuja": 10}
#db.child('Users').child('Profiles').set(data1)


def sendtoDB(name_list1):
    # global data
    user = max(name_list1, key=name_list1.get)
    # data[user] = "Enter"
    data1 = {"Enter": user}
    db.child('Users').child('Enter Notifications').set(data1)
    print("Notification sent to " + user)

while True:
    # grab the frame from the threaded video stream
    frame = vs.read()

    # resize the frame to have a width of 600 pixels (while
    # maintaining the aspect ratio), and then grab the image
    # dimensions

    frame = imutils.resize(frame, width=600)

    (h, w) = frame.shape[:2]

    # construct a blob from the image
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

    # apply OpenCV's deep learning-based face detector to localize
    # faces in the input image
    detector.setInput(imageBlob)
    detections = detector.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections
        if confidence > args["confidence"]:
            # compute the (x, y)-coordinates of the bounding box for
            # the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # extract the face ROI
            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            # ensure the face width and height are sufficiently large
            if fW < 20 or fH < 20:
                continue

            # construct a blob for the face ROI, then pass the blob
            # through our face embedding model to obtain the 128-d
            # quantification of the face
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                             (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            # perform classification to recognize the face
            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]

            # draw the bounding box of the face along with the
            # associated probability
            text = "{}: {:.2f}%".format(name, proba * 100)
            text = name
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          (0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

            if name in name_list.keys():
                name_list[name] += 1
            else:
                name_list[name] = 0

            if name_list[name] > 40:
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 128, 0), 2)
                cv2.putText(frame, text, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 128, 0), 2)
                if name_list[name] > 60:
                    temp1 = 1
                    sendtoDB(name_list)
                    name_list = {}

    if temp3:
        break

    # update the FPS counter
    fps.update()

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break


#vs.release()
vs.stop()
cv2.destroyAllWindows()
