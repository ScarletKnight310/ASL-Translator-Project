import os # for getting data in src dir
import time # time of os / for delays
import cv2 # get data from webcam
import mediapipe as mp # mediapipe extra, analyze data
import numpy as np # number related
import eel
from matplotlib import pyplot as plt # plot / draw / organize data
from scipy import stats # for predictions
import random
import subprocess

from tensorflow.keras.models import Sequential # For Seq Neu Net
import tensorflow as tf # LSTM Layer, for action detection
from tensorflow.keras.layers import LSTM, Dense

eel.init("web") # prep ui for Full Launch
launchFullApp = True

# For data collection and prediction
sequence = []
sentence = []
predictions = []

# May change based on current model
actions = np.array(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'space'])
no_sequences = 222 # number of samples per sign
sequence_length = 28 # Videos are going to be 'n' frames in length

# for mediapipe detections
threshold = 0.5
DetectConfi = 0.5
TrackingConfi = 0.5

# for mediapipe dectection and drawing
cap = cv2.VideoCapture(0)
handsModule = mp.solutions.hands
drawingModule = mp.solutions.drawing_utils

# compiling model
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(sequence_length,63)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# get model path, load weights
path = os.getcwd()
parent = os.path.dirname(path)
ai_path = os.path.join(parent, 'ASL_DLModel.h5')
model.load_weights(ai_path)

# prep image path
video_path = os.path.join(path, "web")
imageName = "video.png"
# Only takes 1 hand, returns array of points based on handlandmarks detected by mediapipe
def extract_hand_keypoints(handLandmarks): # passes a single hand in results.multi_hand_landmarks
    h_data = np.zeros(21*3)
    
    if handLandmarks is not None:
        h_data = np.array([[lm.x, lm.y, lm.z] for lm in handLandmarks.landmark])

    return h_data.flatten()

# detect hands with mediapipe
def mp_hand_detection(image):
    with handsModule.Hands(static_image_mode=True, min_detection_confidence=DetectConfi, min_tracking_confidence=TrackingConfi) as hands:
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return image, results

@eel.expose
def openASLTRlib():
    subprocess.Popen('explorer \"'+os.path.join(parent, 'ASL System Library')+'\"')

# get the current webcam frame / Display image, return prediction
@eel.expose
def get_current_frame(showHandTracking=True): # returns result and image
    global sequence
    global imageName

    res = None
    with handsModule.Hands(static_image_mode=True, min_detection_confidence=DetectConfi, min_tracking_confidence=TrackingConfi) as hands: # Set mediapipe model 
        ret, frame = cap.read()
        image, results = mp_hand_detection(frame)

        # Prediction logic
        if results.multi_hand_landmarks is not None:
            for handLandmarks in results.multi_hand_landmarks:
                keypoints = extract_hand_keypoints(handLandmarks)
                if showHandTracking:
                    drawingModule.draw_landmarks(image, handLandmarks, handsModule.HAND_CONNECTIONS)
                sequence.append(keypoints)
        
        sequence = sequence[-sequence_length:]

        if len(sequence) == sequence_length:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            predictions.append(np.argmax(res))

        cv2.putText(image, str(actions[np.argmax(res)]), (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        try: 
            os.remove(os.path.join(video_path, imageName))
        except: pass
        number = random.randint(1000,9999)
        imageName = "video"+str(number)+".png"
        cv2.imwrite(os.path.join(video_path, imageName), image)

        return [str(actions[np.argmax(res)]), image, imageName]

def webcamDisplay():
    while True:            
        # Show to screen
        resu = get_current_frame()    
        print(resu[0])
        cv2.imshow('ASL-Translator-App', resu[1])

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
def startUI():
    eel.start("index.html")

if __name__ == '__main__':
    if launchFullApp: # Start full app
        # p1 = Process(target = webcamDisplay)
        # p1.start()
        # p2 = Process(target = startUI)
        # p2.start()
        startUI()
    else: # launch lite app / only webcam and terminal output
        webcamDisplay()