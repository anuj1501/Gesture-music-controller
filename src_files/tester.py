import cv2
from tensorflow.keras.models import load_model
from keras_preprocessing import image
import numpy as np
import cv2
from pygame import mixer
import os


IMG_SIZE = 250

song_iter = 0

start_flag = False

next_flag = True

model = load_model('C:/Users/ANUJ/Desktop/gesture-recognition/model/gesture_model.h5')

# Label 
labels = ["unpause","none","pause","next","start","stop"]

#Starting the mixer 
mixer.init()  
  
#list of songs
songs = os.listdir("../songs")

# Setting the volume 
mixer.music.set_volume(0.3)

def rescale_frame(frame, percent=150):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)


def play(iter):

        index = int(song_iter % len(songs))
        
        print(index)
        # Loading the song 
        mixer.music.load("C:/Users/ANUJ/Desktop/gesture-recognition/songs/" + songs[index])

        mixer.music.play()


cap = cv2.VideoCapture(0)


while(True):
    
    _, init_frame = cap.read()

    frame = rescale_frame(init_frame)
    
    # Drawing the ROI
    # The increment/decrement by 1 is to compensate for the bounding box
    cv2.rectangle(frame,(100,100),(550,550),(0,255,0),2)
    # Extracting the ROI
    roi = frame[100:550,100:550]
    
    # Resizing the ROI so it can be fed to the model for prediction
    roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE)) 

    roi_grey = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    #image_grey_final = cv2.threshold(roi_grey,120,255,cv2.THRESH_BINARY)[1]
    image_grey_final = cv2.adaptiveThreshold(roi_grey,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,115,1)

    cv2.imshow('threshold',image_grey_final)
    
    image_array = image.img_to_array(image_grey_final)

    

    image_array = image_array.reshape((1,250,250,1))

    image_array = image_array/255.
    # Batch of 1

    prediction = model.predict(image_array)
    
    result = labels[np.argmax(prediction)]

    if result == 'start' and start_flag==False:
    
        mixer.music.load("C:/Users/ANUJ/Desktop/gesture-recognition/songs/" + songs[0])

        mixer.music.play()
        
        start_flag = True

    elif result == 'pause':

        mixer.music.pause()

    elif result == 'unpause':

        mixer.music.unpause()

    elif result == 'next' and next_flag == False:

        song_iter = song_iter + 1

        play(song_iter)

        next_flag = True

    elif result == 'stop':

        mixer.music.stop()

        next_flag = False


    
    # Displaying the predictions
    cv2.putText(frame, result, (10, 120), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255), 3)

    cv2.imshow('frame',frame)    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

