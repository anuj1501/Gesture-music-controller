# Gesture-music-controller

1. An application that controls music player based on a set of predefined static
hand gestures which is made customizable for the user to add as per his or her choice.

2. Use the Convolutional Neural Networks to recognize the real time gesture and
direct the control using the Pygame library.

3. Created a functionality which can help a person create his own custom dataset, making the application more generalized

4. The model has been trained on GPU power so as to improve the overall training time.

5. Tried to improve the accuracy of the model using few object localization techniques like using the palm detector XML file, or using background cancellation techniques

6. Currently working on the inclusion of real-time detection using 3D CNNs and LSTMs and add more gestures to the dataset to make the overall performance user friendly.
 
7. Worked on deploying the model using the Flask Web API, to render the video screen on a webpage, only shortcoming was the image quality getting deteriorated, yet the overall performance was quite satisfiable.

Salient features include playing the playlist on loop and if no gesture is done, the music either wont start or it wont be played.
