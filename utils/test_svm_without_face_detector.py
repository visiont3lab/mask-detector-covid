import joblib
import cv2
import numpy as np

class_names = ["mask", "no-mask"]

model = joblib.load("../models/mask-classifiers/model.pkl")

# define a video capture object
vid = cv2.VideoCapture(0)
  
while(True):
      
	# Capture the video frame
	# by frame
	ret, frame = vid.read()

	#  BGR 2 RGB
	#im_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	# Resize
	im_np = cv2.resize(frame, (64,64))

	# metto tutto su una riga
	X = im_np.reshape(1,-1)


	Y_hat = model.predict(X)
	idx = np.argmax(Y_hat)
	print(class_names[idx])


	# Display the resulting frame
	cv2.imshow('frame', frame)
	  
	# the 'q' button is set as the
	# quitting button you may use any
	# desired button of your choice
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()


#im_np = cv2.imread( "data/crop_339_img_000464.jpeg.jpg", cv2.IMREAD_COLOR ) 
    

