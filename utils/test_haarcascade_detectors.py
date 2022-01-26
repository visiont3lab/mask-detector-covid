import cv2 
import os
import numpy as np
import joblib 

# Model loaded
model_face = cv2.CascadeClassifier("../models/haar-cascade-files/haarcascade_frontalface_default.xml")
model_eye = cv2.CascadeClassifier("../models/haar-cascade-files/haarcascade_eye.xml")
model = joblib.load("../models/mask-classifiers/model.pkl")


cap = cv2.VideoCapture(0 )

while ( cap.isOpened() ):

    ret, frame = cap.read() # BGR

    if ret:
        #frame = cv2.cvtColor(frameBGR, cv2.COLOR_BGR2RGB)

        draw = frame.copy()
        
        faces = model_face.detectMultiScale(frame,scaleFactor=1.8,minNeighbors=4, flags=cv2.CASCADE_DO_ROUGH_SEARCH | cv2.CASCADE_SCALE_IMAGE)
        for face in faces:
            x,y,w,h = face
            roi = frame[y:y+h,x:x+w]
            cv2.rectangle(draw,(x,y),(x+w,y+h),(255,0,255),1)

            #cv2.imshow("ROI", roi)

            ##########
            # classifier prediction
            # Resize
            im_np = cv2.resize(roi, (64,64))
            # metto tutto su una riga
            X = im_np.reshape(1,-1)
            # prediction
            Y_hat = model.predict_proba(X)   

            res = "no_mask: " + str( round(Y_hat[0][1],2) )
            if Y_hat[0][0] >= 0.5:
                res = "mask: " + str( round(Y_hat[0][0],2) )

            cv2.putText(draw, res , (10, 40), cv2.FONT_HERSHEY_SIMPLEX,  1, (255, 0, 0), 2, cv2.LINE_AA)

            ##########

            eyes = model_eye.detectMultiScale(roi,scaleFactor=1.1,minNeighbors=4, flags=cv2.CASCADE_DO_ROUGH_SEARCH | cv2.CASCADE_SCALE_IMAGE)
            for eye in eyes:
                xx,yy,ww,hh = eye
                cv2.rectangle(draw,(x+xx,y+yy),(x+xx+ww,y+yy+hh),(255,0,0),1)

        cv2.imshow('frame', draw)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
