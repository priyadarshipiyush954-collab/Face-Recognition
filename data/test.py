from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
from win32com.client import Dispatch

def speak(str1):
    speak=Dispatch(("SAPI.SpVoice"))
    speak.Speak(str1)

# --- 1. SETUP CAMERA & DETECTOR (Fixed Path) ---
video=cv2.VideoCapture(0)
# This uses the internal cv2 path so it always finds the file
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# --- 2. LOAD DATA ---
with open('data/names.pkl', 'rb') as w:
    LABELS = pickle.load(w)
with open('data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

print(f'Shape of Faces matrix --> {FACES.shape}')
print(f'Count of Names --> {len(LABELS)}')

# --- 3. AUTO-REPAIR (The Fix for your Error) ---
# This block checks if the counts mismatch and fixes the files permanently
if len(LABELS) != FACES.shape[0]:
    print(f"⚠️ DATA MISMATCH DETECTED! Names: {len(LABELS)}, Faces: {FACES.shape[0]}")
    
    if len(LABELS) > FACES.shape[0]:
        print("-> Fixing: Trimming extra names...")
        LABELS = LABELS[:FACES.shape[0]]
        # Save the fixed list back to file so add_faces.py works correctly next time
        with open('data/names.pkl', 'wb') as w:
            pickle.dump(LABELS, w)
            
    elif FACES.shape[0] > len(LABELS):
        print("-> Fixing: Trimming extra faces...")
        FACES = FACES[:len(LABELS)]
        # Save the fixed faces back to file
        with open('data/faces_data.pkl', 'wb') as f:
            pickle.dump(FACES, f)
            
    print("✅ files synchronized. Resuming...")

# --- 4. TRAIN MODEL ---
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

COL_NAMES = ['NAME', 'TIME']

while True:
    ret,frame=video.read()
    
    # --- FIX 1: Flip the camera so it's not inverted ---
    frame = cv2.flip(frame, 1) 
    
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces=facedetect.detectMultiScale(gray, 1.3 ,5)
    for (x,y,w,h) in faces:
        crop_img=frame[y:y+h, x:x+w, :]
        resized_img=cv2.resize(crop_img, (50,50)).flatten().reshape(1,-1)
        output=knn.predict(resized_img)
        ts=time.time()
        date=datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp=datetime.fromtimestamp(ts).strftime("%H:%M-%S")
        exist=os.path.isfile("Attendance/Attendance_" + date + ".csv")
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
        cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)
        cv2.putText(frame, str(output[0]), (x,y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (50,50,255), 1)
        attendance=[str(output[0]), str(timestamp)]
    
    # --- FIX 2: Show FRAME directly (No Background Image) ---
    cv2.imshow("Frame", frame)
    
    k=cv2.waitKey(1)
    if k==ord('o'):
        speak("Attendance Taken..")
        time.sleep(5)
        if exist:
            with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
                writer=csv.writer(csvfile)
                writer.writerow(attendance)
            csvfile.close()
        else:
            with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
                writer=csv.writer(csvfile)
                writer.writerow(COL_NAMES)
                writer.writerow(attendance)
            csvfile.close()
    if k==ord('q'):
        break
video.release()
cv2.destroyAllWindows()