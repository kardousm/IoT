import os
import pyrebase
import urllib
import imutils

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
#
# def get_path(blob):
#     blob_str = str(blob)
#     blob_str1 = blob_str.split(',')
#     path = blob_str1[1].strip()
#     cleaned_path = path[:-1]
#     return cleaned_path
#
#
# print("Accessing Firebase Storage")
# firebase = pyrebase.initialize_app(config)
# storage = firebase.storage()
# files = storage.list_files()
# blobs = []
#
# for file in files:
#     blobs.append(file)
# del blobs[0]
# print(blobs)

#path = get_path(blobs[1])
#url = storage.child("dataset/Fares/1.jpg").get_url(None)
#img4 = imutils.url_to_image(url)



# print("Downloading Files")
# for blob in blobs:
#     blob_str = str(blob)
#     if blob_str[-2] == '/':
#         count = 1
#         name = blob_str.split('/')[1]
#         cp = '/Users/mariokardous/Desktop/opencv-face-recognition/test'
#         new = cp + '/' + name
#         os.makedirs(new, exist_ok=True)
#     else:
#         updated_name = new + '/' + str(count) + '.jpg'
#         blob.download_to_filename(updated_name)
#         count += 1



os.system("python3 extract_embeddings.py --dataset test --embeddings output/embeddings.pickle --detector face_detection_model --embedding-model openface_nn4.small2.v1.t7")
os.system("python3 train_model.py --embeddings output/embeddings.pickle --recognizer output/recognizer.pickle --le output/le.pickle")
os.system("python3 recognize_video.py --detector face_detection_model --embedding-model openface_nn4.small2.v1.t7 --recognizer output/recognizer.pickle --le output/le.pickle")