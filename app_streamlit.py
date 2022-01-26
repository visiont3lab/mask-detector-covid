import streamlit as st
import cv2 
import numpy as np
import time
import os
import pickle 

class_names = ["mask", "no-mask"]
with open("models/mask-classifiers/model_augmented.pkl", 'rb') as f:
  model_mask = pickle.load(f)


# Windows 
st.title("Setup Configuration")
st.subheader(" Setup")
st.text("Ciao Manuel")
code = '''
    virtualenv env

    Windows: .\env\Scripts\activate
    Linux/Mac: source env/bin/activate

    pip streamlit==0.62 plotly==4.12
    pip install opencv-python
    '''

st.code(code, language='bash')

choice = st.radio(label="", options=["Start","Stop"], index=1)
running=False
if choice=="Start":
    st.subheader("Mask Detector: Started")

    cap = cv2.VideoCapture(0)
    cascade_faces = cv2.CascadeClassifier(os.path.join('models','haar-cascade-files','haarcascade_frontalface_default.xml'))
    
    
    image_placeholder = st.empty()
    running= True
    while(running):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            
            im_color = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            im_gray = cv2.cvtColor(im_color,cv2.COLOR_RGB2GRAY)
            faces = cascade_faces.detectMultiScale(im_gray, 1.5,5,cv2.CASCADE_DO_ROUGH_SEARCH | cv2.CASCADE_SCALE_IMAGE)

            for (x,y,w,h) in faces:
                roi = frame[y:y+h,x:x+w]
                cv2.rectangle(im_color,(x,y),(x+w,y+h),(255,0,0),2)
        
                # -----
                # 64x64x3 --> messe su una linea
                roi = cv2.resize(roi,(64,64))
                inpX = roi.reshape(1,-1)
                y_hat = model_mask.predict_proba(inpX)
                idx = np.argmax(y_hat)
                cv2.putText(im_color, class_names[idx] + " " +  str(np.round(y_hat[0][idx])), (50, 50),  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            
            image_placeholder.image(im_color)
            time.sleep(0.033)
    cap.release()

else:
    # non necessario streamlit riesegue il codice dall'inizio
    running = False
    st.subheader("Mask Detector: Stopped")




# API REFERENCE: https://docs.streamlit.io/en/stable/api.html