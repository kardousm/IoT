from imutils.video import VideoStream
import cv2
import imutils
import numpy
import pyrebase
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import threading
import time

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
cred = credentials.Certificate('maneesh-iot-firebase-adminsdk-m9gqi-c10e487c04.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

# users_ref = db.collection(u'users')
# docs = users_ref.stream()
# for doc in docs:
#     print(doc.to_dict())

# users_red = db.collection(u'Profiles').document('vBelq0WbFAlh5fKeDuiy')
# snapshot = users_red.get()
# print(snapshot.to_dict())


#Find user names
docs = db.collection(u'users').where('name', '==', 'Maneesh').stream()
dict1 = {}
for doc in docs:
    dict1 = doc.to_dict()

def get_notifications(notifications, check_unknown):
    array_temp = []
    if check_unknown:
        for element in notifications:
            if type(element) == str:
                array_temp.append('remind stranger to ' +  element)
            else:
                for element2 in element['customnotifications']:
                    array_temp.append('remind stranger to ' + element2)
        return array_temp

    else:
        for element in notifications:
            if type(element) == str:
                array_temp.append('don\'t forget to ' +  element)
            else:
                for element2 in element['customnotifications']:
                    array_temp.append('don\'t forget to ' + element2)
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
        return [name + ',' + ' ' + notif for notif in notif_array]


array1 = get_notifications(dict1['exitnotifications'], 1)
array1 = update_notifications(array1, 0, 1, 'Mario')
print(array1)






# print(dict1)
# print(dict1['entryNotifications'])
# print(dict1['entryNotifications']['custom'])

#Send notificaitons to Maneesh
# docs = db.collection(u'Notifications').document('t3lanyWN1XDw4n8Vgkpd')
# docs.set({'Notification': ['wear mask', 'social distance']})




