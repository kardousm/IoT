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
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import threading

# config = {
#     "apiKey": "AIzaSyDVvv_NerGXIl7xXVDiZB6hWJoQAb0sZDQ",
#     "authDomain": "test-2c93b.firebaseapp.com",
#     "databaseURL": "https://test-2c93b.firebaseio.com",
#     "projectId": "test-2c93b",
#     "storageBucket": "test-2c93b.appspot.com",
#     "messagingSenderId": "782137670239",
#     "appId": "1:782137670239:web:4e9b9983730397f161f9ea",
#     "measurementId": "G-W7PVLBLNV5",
#     "serviceAccount": "test-2c93b-firebase-adminsdk-fbs0y-24184fce57.json"
# }

#Maneesh
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
track = None
admin = None

def get_notifications(notifications, check_unknown):
    array_temp = []
    if check_unknown:
        for element in notifications:
            if type(element) == str:
                array_temp.append('remind stranger to ' +  element)
        return array_temp

    else:
        if admin:
            for element in notifications:
                if type(element) == str:
                    array_temp.append('don\'t forget to ' +  element)
                else:
                    for element2 in element['customnotifications']:
                        array_temp.append('don\'t forget to ' + element2)
        else:
            for element in notifications:
                if type(element) == str:
                    array_temp.append(element)
                else:
                    for element2 in element['customnotifications']:
                        array_temp.append(element2)
        return array_temp

def update_notifications(notif_array, check_unknown, entry_or_exit, name):
    # name += '{0} '
    # notif_array = [name.format(i) for i in notif_array]
    # return notif_array
    if check_unknown:
        if entry_or_exit == 0:
            return ['A stranger has entered, ' + notif for notif in notif_array]
        else:
            return ['A stranger has exited, ' + notif for notif in notif_array]

    else:
        if admin:
            return [name + ',' + ' ' + notif for notif in notif_array]
        else:
            return ['Please remind ' + name + ' to ' + notif for notif in notif_array]


def ignore_first():
    global temp1
    global temp2
    if temp1 == 0:
        temp2 = False
        temp1 = 1
        return temp2
    else:
        return True


# def stream_handler(post):
#     global temp3
#     if ignore_first():
#         print('Database has been updated. Re-running ML Model.')
#         os.system("python3 extract_embeddings.py --dataset test --embeddings output/embeddings.pickle --detector face_detection_model --embedding-model openface_nn4.small2.v1.t7")
#         os.system("python3 train_model.py --embeddings output/embeddings.pickle --recognizer output/recognizer.pickle --le output/le.pickle")
#         print("New facial recognition window is currently opening")
#         temp3 = True




# firebase = pyrebase.initialize_app(config)
# db = firebase.database()
# my_stream = db.child("Users").child("Profiles").stream(stream_handler, None)
# data2 = {"Exit": "None"}
# db.child('Users').child('Exit Notifications').set(data2)


#Maneesh
firebase = pyrebase.initialize_app(config)
storage = firebase.storage()
cred = credentials.Certificate('maneesh-iot-firebase-adminsdk-m9gqi-c10e487c04.json')
firebase_admin.initialize_app(cred)
db = firestore.client()


# Create an Event for notifying main thread.
callback_done = threading.Event()

#Create a callback on_snapshot function to capture changes
def on_snapshot(doc_snapshot, changes, read_time):
    global temp3
    if ignore_first():
        for doc in doc_snapshot:
            os.system("python3 extract_embeddings.py --dataset test --embeddings output/embeddings.pickle --detector face_detection_model --embedding-model openface_nn4.small2.v1.t7")
            os.system("python3 train_model.py --embeddings output/embeddings.pickle --recognizer output/recognizer.pickle --le output/le.pickle")
            print("New facial recognition window is currently opening")
            temp3 = True
    callback_done.set()

doc_ref = db.collection(u'Profiles').document(u'vBelq0WbFAlh5fKeDuiy')


# Watch the document
doc_watch = doc_ref.on_snapshot(on_snapshot)



def sendtoDB(name_list1, entry_or_exit):
    check_unknown = 0
    # global data
    global admin
    global track
    user = max(name_list1, key=name_list1.get)
    if user == track:
        pass
    else:
        if user == 'Unknown':
            check_unknown = 1
        if entry_or_exit == 0:
            docs = db.collection(u'users').where('name', '==', user).stream()
            dict1 = {}
            for doc in docs:
                dict1 = doc.to_dict()

            admin = dict1['admin']

            temp = get_notifications(dict1['entrynotifications'], check_unknown)
            update_temp = update_notifications(temp, check_unknown, entry_or_exit, user)
            docs = db.collection(u'Notifications').document('t3lanyWN1XDw4n8Vgkpd')
            docs.set({'Notification': update_temp})
            print("Entrance Notification sent regarding " + user)
            track = user
        else:
            docs = db.collection(u'users').where('name', '==', user).stream()
            dict1 = {}
            for doc in docs:
                dict1 = doc.to_dict()

            temp = get_notifications(dict1['exitnotifications'], check_unknown)
            update_temp = update_notifications(temp, check_unknown, entry_or_exit, user)
            docs = db.collection(u'Notifications').document('t3lanyWN1XDw4n8Vgkpd')
            docs.set({'Notification': update_temp})
            print("Exit Notification sent regarding " + user)
            track = user


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
            cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

            if name in name_list.keys():
                name_list[name] += 1
            else:
                name_list[name] = 0

            if name_list[name] > 40:
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 128, 0), 2)
                cv2.putText(frame, text, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 128, 0), 2)
                if name_list[name] > 60:
                    temp1 = 1
                    sendtoDB(name_list, 0)
                    # user = max(name_list, key=name_list1.get)
                    # print(user)
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

os.system("python3 recognize_video.py --detector face_detection_model --embedding-model openface_nn4.small2.v1.t7 --recognizer output/recognizer.pickle --le output/le.pickle")
#vs.release()
vs.stop()
cv2.destroyAllWindows()
