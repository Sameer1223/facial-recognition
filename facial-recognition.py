#Happiness Classifier

#Importing Libraries
import cv2

#Loading the cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

#Function that will detect features
def detect(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    #Looping through the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
        roi_gray = gray[y:y+h, x:x+w] #Looking for the gray zone 
        roi_color = frame[y:y+h, x:x+w] #Looking for the color zone
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 22)
        for (ex, ey, ew, eh) in eyes: #Looking for eyes in the face zone
            cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.7, 22)
        for (sx, sy, sw, sh) in smiles: #Looking for eyes in the face zone
            cv2.rectangle(roi_color, (sx,sy), (sx+sw, sy+sh), (0,0,255), 2)
    return frame

#Face Recognition With Webcam
video_capture = cv2.VideoCapture(0)
while True:
    _, frame = video_capture.read() #VidCapture gives two arguments, ignore first one
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Convert to gray
    canvas = detect(gray, frame) #Apply detect function
    cv2.imshow('Video', canvas) #Show the detected rectangles on vid
    if cv2.waitKey(1) and 0xFF == ord('q'): #If q is pressed stop detecting
        break
#Turning off the webcam
video_capture.release()
cv2.destroyAllWindows()    