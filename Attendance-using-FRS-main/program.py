import face_recognition
# pip install dlib
# pip install face_recognition
#The face_recognition library is an open-source Python tool to perform facial recognition. Widely regarded for its simpleness and accuracy.
import cv2 
# OpenCV (Open Source Computer Vision Library) is a massive open-source software library used for computer vision, machine learning, and image processing. It is written natively in C++.It contains over 2,500 optimized algorithms that allow computers to "see" and interpret images and videos much like a human eye.
import os
# os is a built-in module in python,it allows your program to interact with the Operating System.it is primarily used for automated file management.
# os.listdir(): Lists all files inside a specific folder.
# os.path.join(): Connects a folder name and a file name into a full path.

import numpy as np
#A NumPy (Numerical Python)is the foundational library, its primary role is to handle multidimensional arrays. NumPy is the "math engine"
import csv
# CSV stands for Comma-Separated Values. It is a simple, plain-text file format used to store data in a tabular (table-like) structure.
# The csv module is typically used to create a permanent log of who was recognized and at what time.
from datetime import datetime

video_capture = cv2.VideoCapture(0)
#The code video_capture = cv2.VideoCapture(0) means you are creating a VideoCapture object named video_capture that connects to and starts capturing video from the default camera (webcam) of your computer. 
#VideoCapture(): This is a class method within the OpenCV library used to open a video file or a device for video capturing.
#Here, 0 typically refers to the primary or default camera (e.g., a built-in laptop webcam).
#If you have multiple cameras, 1 would refer to the second camera, 2 to the third, and so on.

jobs_image = face_recognition.load_image_file("jobs.jpg")
jobs_encoding = face_recognition.face_encodings(jobs_image)[0]  
#[0]: Because the function returns a list, the [0] selects the first face detected in that image

ratan_tata_image = face_recognition.load_image_file("ratan_tata.jpg")
ratan_tata_encoding = face_recognition.face_encodings(ratan_tata_image)[0]
#The code ratan_tata_encoding = face_recognition.face_encodings(ratan_tata_image)[0] is using the face_recognition Python library to generate a 128-dimensional facial embedding (or "encoding") of Ratan Tata's face from a loaded image. 
# The face_recognition library scans the image and returns a list of every face it finds. If there is only one person in the photo, the list will have exactly one item at index [0].

sadmona_image = face_recognition.load_image_file("sadmona.jpg")
sadmona_image_encoding = face_recognition.face_encodings(sadmona_image)[0]

tesla_image = face_recognition.load_image_file("tesla.jpg")
tesla_image_encoding = face_recognition.face_encodings(tesla_image)[0]

known_face_encodings = [
    jobs_encoding,
    ratan_tata_encoding,
    sadmona_image_encoding,
    tesla_image_encoding
]

known_face_names = [
    "Jobs",
    "Ratan Tata",
    "Sadmona",
    "Tesla"
]

students = known_face_names.copy()

face_locations = []
face_encodings = []
face_names = []
s=True


now = datetime.now()
current_date = now.strftime("%Y-%m-%d")
#The line of code current_date = now.strftime("%Y-%m-%d") means the current date and time stored in the variable now is being formatted into a specific string format and assigned to a new variable called current_date
#strftime(): This is a method that converts a datetime object into a formatted string. The "str" stands for string, and "f" for format.



f =open(current_date+'.csv','w+',newline='')
lnwriter = csv.writer(f)
# lnwriter: This is a variable that stores a writer object.
# csv.writer(f): This function from the csv module creates a writer object that is responsible for converting your data (e.g., a list of strings or numbers) into the properly formatted, delimited string structure required for a CSV file. The f file object is passed to it so the writer knows where to output the formatted data. 


while True:
    _, frame = video_capture.read()
#When you call video_capture.read(), it performs two primary actions simultaneously:
#Grabs and Decodes: It grabs the next available frame from the video source (like your webcam or a video file) and decodes it into a usable image format.
#Returns a Tuple: It always returns a pair of values.
# The first value is a True or False flag.
# True: A frame was successfully captured and decoded. False: No frame was grabbed.
# The second value is the actual image data stored as a NumPy array.This array contains the pixel information (BGR format) for that specific moment in time.

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    #This shrinks the image to 25% of its original size (1/4 scale).
    rgb_small_frame = small_frame[:, :, ::-1]
#In Python’s NumPy library, this syntax is a slicing trick used to reverse the order of the color channels.
# Specifically, it converts an image from BGR (Blue, Green, Red) to RGB (Red, Green, Blue).
# An image array has three dimensions: [Height, Width, Color Channels].
# First Colon (:): Selects all rows (the entire height of the image).
# Second Colon (, :): Selects all columns (the entire width of the image).
# The Step Slice (, ::-1): Selects the color channels but in reverse order.
# Instead of index 0, 1, 2 (Blue, Green, Red), it takes 2, 1, 0 (Red, Green, Blue)
    if s:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = ""
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                
            face_names.append(name)
            if name in known_face_names:
                if name in students:
                    students.remove(name)
                    print(students)
                    current_time = now.strftime("%H:%M:%S") 
                    lnwriter.writerow([name,current_time])
    
    cv2.imshow("attendance", frame) 
#In Python's OpenCV library (cv2), this command is used to display an image or video frame in a graphical window. 
#"attendance": This is the window name (a string). It appears in the title bar of the pop-up window.
# frame: This is the image data (a NumPy array) that you want to show. In an attendance system, this is usually the current live image captured from your webcam
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Wait 1 millisecond for a key press; if 'q' is pressed, break the loop
    
video_capture.release()
cv2.destroyAllWindows()
f.close()             