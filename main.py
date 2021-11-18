
import cv2
import numpy as np
import pickle 

def rotate(image, angle, center=None, scale=1.0):
    # grab the dimensions of the image
    (h, w) = image.shape[:2]

    # if the center is None, initialize it as the center of
    # the image
    if center is None:
        center = (w // 2, h // 2)

    # perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    # return the rotated image
    return rotated

def  findLargestBB(bbs):
  areas = [w*h for x,y,w,h in bbs]
  if not areas:
      return False, None
  else:
      i_biggest = np.argmax(areas) 
      biggest = bbs[i_biggest]
      return True, biggest

cap = cv2.VideoCapture(0) #"data/video.mp4")

if (cap.isOpened()== False): 
  print("Error opening video stream or file")

model_face = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

# Load mask classifier
with open("models/model.pkl", 'rb') as f:
  model_mask = pickle.load(f)

class_names = ["mask", "no-mask"]

while(cap.isOpened()):

  # lettura immagine
  ret, frame = cap.read()  #BGR
  
  # coversione immagine da BGR a RGB
  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
  
  # Rotazione dell'immagine
  #frame = rotate(frame, -90)

  # Trova tutte le facce nell'immagine
  faces = model_face.detectMultiScale(frame,scaleFactor=1.8,minNeighbors=4, flags=cv2.CASCADE_DO_ROUGH_SEARCH | cv2.CASCADE_SCALE_IMAGE)
  
  # Trova la faccia piu grande (area piu grande)
  #ret, facebig = findLargestBB(faces)

  for (x,y,w,h) in faces:
  # Per ogni faccia fai qualcosa
  #if ret == True:

    # Extra coordiante of largest image
    #x,y,w,h = facebig
    
    # Crop image 
    roi = frame[y:y+h,x:x+w]

    # -----
    # 64x64x3 --> messe su una linea
    roi = cv2.resize(roi,(64,64))
    inpX = roi.reshape(1,-1)
    y_hat = model_mask.predict_proba(inpX)
    idx = np.argmax(y_hat)
    cv2.putText(frame, class_names[idx] + " " +  str(np.round(y_hat[0][idx])), (50, 50),  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
   
    #print(y_hat)
    #print(int(y_hat))
    #y_hat = model_mask.predict(inpX)
    #cv2.putText(frame, class_names[int(y_hat[0])], (50, 50),  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
  
    #print(y_hat)
    # -----
    
    

    # Disegna il quadrato
    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

    # Solo sul vostro pc
    #cv2.imshow("Roi", roi)

  frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
  cv2.imshow("Image", frame)
  cv2.waitKey(33)

    # In colab o jupyter

    #frame = cv2.resize(frame, (128,128))
    #im_pil = Image.fromarray(roi)
    #display(im_pil)
