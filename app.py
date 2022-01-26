import streamlit as st
import cv2
import numpy as np
import os
import pickle

cascade_faces = cv2.CascadeClassifier(os.path.join('models','haar-cascade-files','haarcascade_frontalface_default.xml'))
with open("models/mask-classifiers/model_augmented.pkl", 'rb') as f:
  model_mask = pickle.load(f)
class_names = ["mask", "no-mask"]  

st.set_option('deprecation.showfileUploaderEncoding', False)

st.title("Face detector")


st.sidebar.markdown('''Parametri''')
scaleFactor = st.sidebar.slider('scaleFactor', 1.0, 2.0, 1.5)
minNeigh = st.sidebar.slider('scaleFactor', 3, 10, 4)

uploaded_file = st.sidebar.file_uploader("Upload Image", type=["png","jpeg","jpg","bmp"])
if uploaded_file is not None:
    #print(np.fromstring(uploaded_file.read(), np.uint8))
    image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8),cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #base64_img_bytes = uploaded_file.read() # byte
    #decoded_image_data = base64.decodebytes(base64_img_bytes)
    #nparr = np.fromstring(decoded_image_data, np.uint8)
    #img = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # cv2.IMREAD_COLOR in OpenCV

    im_gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    faces = cascade_faces.detectMultiScale(im_gray, scaleFactor,minNeigh,cv2.CASCADE_DO_ROUGH_SEARCH | cv2.CASCADE_SCALE_IMAGE)

    for (x,y,w,h) in faces:
        roi = image[y:y+h,x:x+w]
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
        
        roi = cv2.resize(roi,(64,64))
        inpX = roi.reshape(1,-1)
        y_hat = model_mask.predict_proba(inpX)
        idx = np.argmax(y_hat)
        cv2.putText(image, class_names[idx] + " " +  str(np.round(y_hat[0][idx])), (50, 50),  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    st.image(image, use_column_width=True ) # width=700)


