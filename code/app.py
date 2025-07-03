import sqlite3  
from flask import Flask, render_template, Response, request,jsonify
import cv2
import numpy as np
import mediapipe as mp 
import pickle
import nltk
import time
import speech_recognition as sr
import os
import pandas as pd
from playsound import playsound
from threading import Thread




app = Flask(__name__)

mp_drawing = mp.solutions.drawing_utils 
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()
holistic = mp_holistic.Holistic()

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

detected_letters = []
current_letter_duration = 0
min_continuous_duration = 30

@app.route('/')
def index():
    return render_template('index.html')

count = 0
alarm_on = False
alarm_sound = "data/alarm.mp3"

def start_alarm():
    playsound(alarm_sound)


def trigger_action(letter):
    if letter == 'fan on':
        print('Fan On')
        SerialObj = serial.Serial('COM3')
        SerialObj.baudrate = 9600
        SerialObj.bytesize = 8
        SerialObj.parity   ='N'
        SerialObj.stopbits = 1
        SerialObj.write(b'a')
        print('data moved')
        SerialObj.close()
    elif letter == 'fan off':
        print('Fan Off')
        SerialObj = serial.Serial('COM3')
        SerialObj.baudrate = 9600
        SerialObj.bytesize = 8
        SerialObj.parity   ='N'
        SerialObj.stopbits = 1
        SerialObj.write(b'b')
        print('data moved')
        SerialObj.close()
    elif letter == 'light on':
        print('Light On')
        SerialObj = serial.Serial('COM3')
        SerialObj.baudrate = 9600
        SerialObj.bytesize = 8
        SerialObj.parity   ='N'
        SerialObj.stopbits = 1
        SerialObj.write(b'c')
        print('data moved')
        SerialObj.close()
    elif letter == 'light off':
        print('Light Off')
        SerialObj = serial.Serial('COM3')
        SerialObj.baudrate = 9600
        SerialObj.bytesize = 8
        SerialObj.parity   ='N'
        SerialObj.stopbits = 1
        SerialObj.write(b'd')
        print('data moved')
        SerialObj.close()
    elif letter == 'good morning':
        print('Light On')
        SerialObj = serial.Serial('COM3')
        SerialObj.baudrate = 9600
        SerialObj.bytesize = 8
        SerialObj.parity   ='N'
        SerialObj.stopbits = 1
        SerialObj.write(b'c')
        print('data moved')
        SerialObj.close()
    elif letter == 'good night':
        print('Light Off')
        SerialObj = serial.Serial('COM3')
        SerialObj.baudrate = 9600
        SerialObj.bytesize = 8
        SerialObj.parity   ='N'
        SerialObj.stopbits = 1
        SerialObj.write(b'c')
        print('data moved')
        SerialObj.close()
    elif letter == 'alert':
            if not alarm_on:
                alarm_on = True
                t = Thread(target=start_alarm)
                t.daemon = True
                t.start()





@app.route('/generate_frames', methods=['GET', 'POST'])
def generate_frames():
    global body_language_class
  
    body_language_class = None
    consecutive_count = 0
    required_count = 10  

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return render_template("index.html")

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            

            if not ret:
                print("Error: Could not read frame from the camera.")
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            results = holistic.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                       mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                       mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                                       )

            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                       mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                       mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                                       )
            body_language_class = None

           
        

            if results.left_hand_landmarks:
                    pose = results.left_hand_landmarks.landmark
                    pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

                    row = pose_row

                    X = pd.DataFrame([row])
                    body_language_class = model.predict(X)[0]

                    current_detected_letter = body_language_class

                    if current_detected_letter == body_language_class:
                        consecutive_count += 1
                        if consecutive_count == required_count:
                            trigger_action(current_detected_letter)
                            consecutive_count = 0  
                    else:
                        body_language_class = current_detected_letter
                        consecutive_count = 1                            
                                         
           
          
            
                        
            
            
            if  cv2.waitKey(10) & 0xFF == ord('q'):
                break

            


            cv2.putText(image, 'Detection', (95, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (139, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, body_language_class, (90, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (139, 0, 0), 2, cv2.LINE_AA)


            cv2.imshow("Detect Sign",image)
           

    cap.release()
    cv2.destroyAllWindows()
    return render_template("index.html")


       
r = sr.Recognizer()
mic = sr.Microphone()



def speak1():
    with mic as audio_file:
        print("Speak Now...")
        r.adjust_for_ambient_noise(audio_file)
        audio = r.listen(audio_file)
        print("Converting Speech to Text...")
        text= r.recognize_google(audio)
        text=text.lower()
        return text

@app.route('/speak', methods=['GET','POST'])
def speak():
    speech=speak1()
    trigger_action(speech)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=False,port=450)
