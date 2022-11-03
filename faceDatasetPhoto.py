'''
From https://github.com/Mjrovai/OpenCV-Face-Recognition

    ==> Procces face from image file fro training
    ==> face will be stored in "dataset/" directory
    ==> face will have a unique integer id
'''

import cv2
import os

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
currId = len([name for name in os.listdir('database') if name != 'representations_facenet512.pkl'])

# input for file name
photo = input("Insert file name: ")

# For each person, enter one numeric face id
user_name = input('enter user name and press enter: ')
face_id = currId + 1
# user_id = (int(face_id), user_name)

print("\n [INFO] Processing photo ...")

# create 'dataset' directory
if 'database' in os.listdir(os.getcwd()):
    pass
else:
    os.mkdir(os.path.join(os.getcwd(), 'dataset'))

img = cv2.imread(photo)
    cv2.imwrite(f"dataset/{str(face_id)}." + user_name + ".jpg", gray[y:y+h,x:x+w])

print("\n [INFO] Photo saved.")