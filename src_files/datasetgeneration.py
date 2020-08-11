import cv2
import numpy as np

def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

camera = cv2.VideoCapture(0)

count = 900

while(True):

    #read a new frame
    _,fr = camera.read()

    frame = rescale_frame(fr, percent=150)

    cv2.rectangle(frame,(100,100),(550,550),(0,255,0),2)

    cv2.imshow("Capturing Frames",frame)


    key = cv2.waitKey(1)

    if key & 0xFF == ord("q"):
        break
    elif key & 0xFF == ord("z"):

        roi = frame[100:550,100:550]

        resized_image = cv2.resize(roi,(250,250),interpolation = cv2.INTER_AREA)

        image_grey = cv2.cvtColor(resized_image,cv2.COLOR_BGR2GRAY)

        image_grey_final = cv2.threshold(image_grey,90,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]

        cv2.imwrite("dataset/none/{}.jpg".format(count),image_grey_final)

        count = count + 1

        print("saving image {}.jpg to folder".format(count))


camera.release()
cv2.destroyAllWindows()

'''
desc = Script to gather data images with a particular label.
Usage: python gather_images.py <label_name> <num_samples>
The script will collect <num_samples> number of images and store them
in its own directory.
Only the portion of the image within the box displayed
will be captured and stored.
Press 'a' to start/pause the image collecting process.
Press 'q' to quit.

import cv2
import os
import sys

try:
    label_name = sys.argv[1]
    num_samples = int(sys.argv[2])
except:
    print("Arguments missing.")
    print(desc)
    exit(-1)

IMG_SAVE_PATH = 'image_data'
IMG_CLASS_PATH = os.path.join(IMG_SAVE_PATH, label_name)

try:
    os.mkdir(IMG_SAVE_PATH)
except FileExistsError:
    pass
try:
    os.mkdir(IMG_CLASS_PATH)
except FileExistsError:
    print("{} directory already exists.".format(IMG_CLASS_PATH))
    print("All images gathered will be saved along with existing items in this folder")

cap = cv2.VideoCapture(0)

start = False
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    if count == num_samples:
        break

    cv2.rectangle(frame, (100, 100), (500, 500), (255, 255, 255), 2)

    if start:
        roi = frame[100:500, 100:500]
        save_path = os.path.join(IMG_CLASS_PATH, '{}.jpg'.format(count + 1))
        cv2.imwrite(save_path, roi)
        count += 1

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "Collecting {}".format(count),
            (5, 50), font, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Collecting images", frame)

    k = cv2.waitKey(10)
    if k == ord('a'):
        start = not start

    if k == ord('q'):
        break

print("\n{} image(s) saved to {}".format(count, IMG_CLASS_PATH))
cap.release()
cv2.destroyAllWindows()

1. display info when we click on that dot
2. hover over heat map to display total no.


'''