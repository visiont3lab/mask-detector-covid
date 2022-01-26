import joblib
import cv2
import numpy as np

class_names = ["mask", "no-mask"]

model = joblib.load("../models/mask-classifiers/model.pkl")

im_np = cv2.imread( "../dataset/mask/download.jpg", cv2.IMREAD_COLOR ) 
    
#  BGR 2 RGB
#im_np = cv2.cvtColor(im_np, cv2.COLOR_BGR2RGB)

# Resize
im_np = cv2.resize(im_np, (64,64))

# metto tutto su una riga
X = im_np.reshape(1,-1)


Y_hat = model.predict(X)
idx = np.argmax(Y_hat)

print(class_names[idx])




