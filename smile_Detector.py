import cv2

#Face classifier
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_detector = cv2.CascadeClassifier('smile.xml')

#Grab Webcam feed
webcam = cv2.VideoCapture(0)

#Show the current frame
while True:
    #Read current frame from webcam
    successful_frame, frame = webcam.read()

    # If there's an error, abort
    if not successful_frame:
        break

    #Change to grayscale
    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Detect faces first
    faces = face_detector.detectMultiScale(frame_grayscale)
    

    #Run smile detection within each of those faces
    for (x,y,w,h) in faces:
        #Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 200, 50), 4)
        
        the_face= frame[y:y+h, x:x+w]

        #Change to grayscale
        face_grayscale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)

        smiles = smile_detector.detectMultiScale(face_grayscale, scaleFactor=1.7, minNeighbors=20)

        # #Find all smiles in faces
        # for (x_, y_, w_, h_) in smiles:
        #     # Draw a rectangle around the face
        #     cv2.rectangle(the_face, (x_, y_), (x_+w_, y_+h_), (50, 50, 200), 4)

        
        #Label this face as smiling
        if len(smiles) > 0:
            cv2.putText(frame, 'smiling', (x, y+h+40), fontScale=4, fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255, 255, 255))


    

    #show the current frame
    cv2.imshow('Smile Detector', frame)

    #Display
    key=cv2.waitKey(1)

    #Stop if Q key is pressed
    if key==81 or key==113:
        break

#CleanUP    
webcam.release()
cv2.destroyAllWindows()


print("completed")