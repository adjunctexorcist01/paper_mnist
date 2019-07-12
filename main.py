import cv2
import numpy as np
from keras.models import load_model
import os
from functions import x_cord_contour, makeSquare, resize_to_pixel


MODEL_DIR = os.getcwd() + "/models"
PATH_DIR = os.getcwd() + "/test_images/"

clf = load_model(MODEL_DIR+"/emnist_digits_20ep.h5")

def load_predict(clf, PATH_DIR):
	image = cv2.imread(PATH_DIR + "image.jpg", 0)
	text = predict(clf, image)
	return text



def predict(clf, image):
	blur = cv2.GaussianBlur(image, (5, 5), 0)

	#Detecting Canny edges
	canny = cv2.Canny(blur, 30, 150)
	# canny = cv2.bitwise_not(canny)

	contours, _ = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	#Sort out contours left to right by using their x cordinates
	contours = sorted(contours, key = x_cord_contour, reverse = False)

	text_retrieved = []
	# loop over the contours
	for num, c in enumerate(contours):
	    # compute the bounding box for the rectangle
	    (x, y, w, h) = cv2.boundingRect(c)    

	    if w >= 5 and h >= 5:
	        roi = blur[y:y + h, x:x + w]
	        ret, roi = cv2.threshold(roi, 127, 255,cv2.THRESH_BINARY_INV)
	        roi = makeSquare(roi)
	        # roi = cv2.bitwise_not(roi)
	        # cv2.imwrite(f"test{num}.jpg", roi)

	        roi = resize_to_pixel(28, roi)
	        cv2.imshow("ROI", roi)
	        roi = roi / 255.0       
	        roi = roi.reshape(1,28,28,1) 

	        ## Get Prediction
	        res = str(clf.predict_classes(roi, 1, verbose = 1)[0])
	        text_retrieved.append(res)
	        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
	        cv2.putText(image, res, (x , y + 200), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
	        cv2.imshow("image", image)
	        cv2.waitKey(0)
	cv2.destroyAllWindows()
	print("\nThe text is "+" ".join(text_retrieved))

	return text_retrieved


if __name__=="__main__":
	load_predict(clf, PATH_DIR)