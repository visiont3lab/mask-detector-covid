
import numpy as np
import imutils
import time
import cv2
import os

class myCustomDetector:

    def __init__(self):
        # initialize the list of class labels MobileNet SSD was trained to
        # detect, then generate a set of bounding box colors for each class
        self.CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                "sofa", "train", "tvmonitor"]
                
        self.COLORS = np.random.uniform(0, 255, size=(len(self.CLASSES), 3))

        # load our serialized model from disk
        print("[INFO] loading model...")
        self.net = cv2.dnn.readNetFromCaffe(os.path.join('..','models','mobilenetssd','MobileNetSSD_deploy.prototxt.txt'),os.path.join('..','models','mobilenetssd','MobileNetSSD_deploy.caffemodel'))

    def doDetection(self,frame_input):
        res = False
        frame = imutils.resize(frame_input, width=400)

        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

        # pass the blob through the network and obtain the detections and predictions
        self.net.setInput(blob)
        detections = self.net.forward()

        # loop over the detections
        percentage = 0
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > 0.4:
                # extract the index of the class label from the
                # `detections`, then compute the (x, y)-coordinates of
                # the bounding box for the object
                idx = int(detections[0, 0, i, 1])
                if self.CLASSES[idx]=="person":
                    res = True
                    percentage = np.round(confidence*100,3)
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # draw the prediction on the frame
                    label = "{}: {:.2f}%".format(self.CLASSES[idx],
                                                confidence * 100)
                    cv2.rectangle(frame, (startX, startY), (endX, endY),
                                self.COLORS[idx], 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(frame, label, (startX, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS[idx], 2)
            
        return res, frame, percentage


def from_folder(folder_path = "../dataset/mask"):

    dt = myCustomDetector()

    names  = os.listdir(folder_path)
    for name in names:
        filename = os.path.join(folder_path, name)
        print(filename)
        image = cv2.imread(filename,1)
        res, frame, percentage = dt.doDetection(image)

        cv2.imshow("im", frame)
        cv2.waitKey(33)

def from_camera():
    cap = cv2.VideoCapture(0 )

    dt = myCustomDetector()
    while ( cap.isOpened() ):

        ret, frame = cap.read() # BGR

        if ret:
            #frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            res, frame, percentage = dt.doDetection(frame)

            cv2.imshow("im", frame)
            cv2.waitKey(33)

#from_folder()
from_camera()
