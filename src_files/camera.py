import cv2
from tensorflow.keras.models import load_model
from keras_preprocessing import image
import numpy as np
#from pygame import mixer 
  


def predictor(frame):

    IMG_SIZE = 250

    model = load_model('gesture_model.h5')

    # Label 
    labels = ["unpause","none","pause","prev","start","stop"]

    
    # Drawing the ROI
    # The increment/decrement by 1 is to compensate for the bounding box
    cv2.rectangle(frame,(100,100),(550,550),(0,255,0),2)
    # Extracting the ROI
    roi = frame[100:550,100:550]
    
    # Resizing the ROI so it can be fed to the model for prediction
    roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE)) 

    roi_grey = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    image_grey_final = cv2.threshold(roi_grey,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]

    image_array = image.img_to_array(image_grey_final)

    image_array = image_array.reshape((1,250,250,1))

    image_array = image_array/255.
    # Batch of 1

    prediction = model.predict(image_array)
    
    result = labels[np.argmax(prediction)]
    
    # Displaying the predictions
    cv2.putText(frame, result, (10, 120), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255), 3)    

    return frame


def rescale_frame(frame, percent=150):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

        
class Camera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()        

    def get_frame(self):
        _, ini_frame = self.video.read()

        frame = rescale_frame(ini_frame,percent=150)

        # DO WHAT YOU WANT WITH TENSORFLOW / KERAS AND OPENCV
        final_frame = predictor(frame)

        _, jpeg = cv2.imencode('.jpg', final_frame)

        return jpeg.tobytes()