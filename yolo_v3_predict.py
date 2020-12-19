#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 15:39:51 2020

@author: vcroopana
"""


# load yolov3 model and perform object detection
# based on https://github.com/experiencor/keras-yolo3
# ref: https://machinelearningmastery.com/how-to-perform-object-detection-with-yolov3-in-keras/
import numpy as np
from numpy import expand_dims
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from pathlib import Path

class BoundBox:
	def __init__(self, xmin, ymin, xmax, ymax, objness = None, classes = None):
		self.xmin = xmin
		self.ymin = ymin
		self.xmax = xmax
		self.ymax = ymax
		self.objness = objness
		self.classes = classes
		self.label = -1
		self.score = -1

	def get_label(self):
		if self.label == -1:
			self.label = np.argmax(self.classes)

		return self.label

	def get_score(self):
		if self.score == -1:
			self.score = self.classes[self.get_label()]

		return self.score
    
	def getCoordinates(self):
		coords = np.zeros((4,2))
		coords[0,0] = self.xmin # top left
		coords[0,1] = self.ymin
		coords[1,0] = self.xmin # bottom left
		coords[1,1] = self.ymax
		coords[2,0] = self.xmax # bottom right
		coords[2,1] = self.ymax
		coords[3,0] = self.xmax # top right
		coords[3,1] = self.ymin
		coordsStr = " top left:"+ str(self.xmin)+ ","+str(self.ymin)\
        + "\t top right:"+ str(self.xmax) +","+ str(self.ymin)\
        + "\n bottom left:"+ str(self.xmin)+","+ str(self.ymax)\
        + "\t bottom right:"+ str(self.xmax)+","+ str(self.ymax)
		
		return coords

def _sigmoid(x):
	return 1. / (1. + np.exp(-x))

def decode_netout(netout, anchors, obj_thresh, net_h, net_w):
	grid_h, grid_w = netout.shape[:2]
	nb_box = 3
	netout = netout.reshape((grid_h, grid_w, nb_box, -1))
	nb_class = netout.shape[-1] - 5
	boxes = []
	netout[..., :2]  = _sigmoid(netout[..., :2])
	netout[..., 4:]  = _sigmoid(netout[..., 4:])
	netout[..., 5:]  = netout[..., 4][..., np.newaxis] * netout[..., 5:]
	netout[..., 5:] *= netout[..., 5:] > obj_thresh

	for i in range(grid_h*grid_w):
		row = i / grid_w
		col = i % grid_w
		for b in range(nb_box):
			# 4th element is objectness score
			objectness = netout[int(row)][int(col)][b][4]
			if(objectness.all() <= obj_thresh): continue
			# first 4 elements are x, y, w, and h
			x, y, w, h = netout[int(row)][int(col)][b][:4]
			x = (col + x) / grid_w # center position, unit: image width
			y = (row + y) / grid_h # center position, unit: image height
			w = anchors[2 * b + 0] * np.exp(w) / net_w # unit: image width
			h = anchors[2 * b + 1] * np.exp(h) / net_h # unit: image height
			# last elements are class probabilities
			classes = netout[int(row)][col][b][5:]
			box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, objectness, classes)
			boxes.append(box)
	return boxes

def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
	new_w, new_h = net_w, net_h
	for i in range(len(boxes)):
		x_offset, x_scale = (net_w - new_w)/2./net_w, float(new_w)/net_w
		y_offset, y_scale = (net_h - new_h)/2./net_h, float(new_h)/net_h
		boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
		boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
		boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
		boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)

def _interval_overlap(interval_a, interval_b):
	x1, x2 = interval_a
	x3, x4 = interval_b
	if x3 < x1:
		if x4 < x1:
			return 0
		else:
			return min(x2,x4) - x1
	else:
		if x2 < x3:
			 return 0
		else:
			return min(x2,x4) - x3

def bbox_iou(box1, box2):
	intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
	intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
	intersect = intersect_w * intersect_h
	w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
	w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
	union = w1*h1 + w2*h2 - intersect
	return float(intersect) / union

def do_nms(boxes, nms_thresh):
	if len(boxes) > 0:
		nb_class = len(boxes[0].classes)
	else:
		return
	for c in range(nb_class):
		sorted_indices = np.argsort([-box.classes[c] for box in boxes])
		for i in range(len(sorted_indices)):
			index_i = sorted_indices[i]
			if boxes[index_i].classes[c] == 0: continue
			for j in range(i+1, len(sorted_indices)):
				index_j = sorted_indices[j]
				if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
					boxes[index_j].classes[c] = 0

# load and prepare an image
def load_image_pixels(filename, shape):
	# load the image to get its shape
	image = load_img(filename)
	width, height = image.size
	# load the image with the required size
	image = load_img(filename, target_size=shape)
	# convert to numpy array
	image = img_to_array(image)
	# scale pixel values to [0, 1]
	image = image.astype('float32')
	image /= 255.0
	# add a dimension so that we have one sample
	image = expand_dims(image, 0)
	return image, width, height

# get all of the results above a threshold and class label is person
def get_boxes(boxes, labels, thresh):
	v_boxes, v_labels, v_scores = list(), list(), list()
	# enumerate all boxes
	for box in boxes:
		# enumerate all possible labels
		for i in range(len(labels)):
			# check if the threshold for this label is high enough
			if box.classes[i] > thresh and labels[i] == 'person':
				v_boxes.append(box)
				v_labels.append(labels[i])
				v_scores.append(box.classes[i]*100)
				# don't break, many labels may trigger for one box
	return v_boxes, v_labels, v_scores

# draw all results
def draw_boxes(filename, v_boxes, v_labels, v_scores):
	# load the image
	print('filename at draw boxes:'+ str(filename))
	data = pyplot.imread(str(filename))
	# plot the image
	pyplot.imshow(data)
	# get the context for drawing boxes
	ax = pyplot.gca()
	# plot each box
	for i in range(len(v_boxes)):
		box = v_boxes[i]
		# get coordinates
		y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
		# calculate width and height of the box
		width, height = x2 - x1, y2 - y1
		# create the shape
		rect = Rectangle((x1, y1), width, height, fill=False, color='white')
		# draw the box
		ax.add_patch(rect)
		# draw text and score in top left corner
		label = "%s (%.3f)" % (v_labels[i], v_scores[i])
		pyplot.text(x1, y1, label, color='white')
        
#     for ((xA, yA), (xB, yB), color) in zip(refCoords, objCoords, colors):
# 		# draw circles corresponding to the current points and
# 		# connect them with a line
# 		cv2.circle(orig, (int(xA), int(yA)), 5, color, -1)
# 		cv2.circle(orig, (int(xB), int(yB)), 5, color, -1)
# 		cv2.line(orig, (int(xA), int(yA)), (int(xB), int(yB)),
# 			color, 2)
# 		# compute the Euclidean distance between the coordinates,
# 		# and then convert the distance in pixels to distance in
# 		# units
# 		D = dist.euclidean((xA, yA), (xB, yB)) / refObj[2]
# 		(mX, mY) = midpoint((xA, yA), (xB, yB))
# 		cv2.putText(orig, "{:.1f}in".format(D), (int(mX), int(mY - 10)),
# 			cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
# 		# show the output image
# 		cv2.imshow("Image", orig)
# 		cv2.waitKey(0)
	# show the plot
	pyplot.show()

def getYOLOBBox(model, photo_filename, target_w, target_h, draw_boxes_flag, class_threshold=0.6):
    # load yolov3 model
    # model = load_model('yolo_v3_keras/model.h5')
    
    labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
        	"boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        	"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
        	"backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
        	"sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        	"tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
        	"apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
        	"chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
        	"remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
        	"book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
    anchors = [[116,90, 156,198, 373,326], [30,61, 62,45, 59,119], [10,13, 16,30, 33,23]] # define the anchors
    # class_threshold = 0.6    # define the probability threshold for detected objects


    image, image_w, image_h = load_image_pixels(photo_filename, (target_w, target_h))
    yhat = model.predict(image)
        # summarize the shape of the list of arrays
    # print([a.shape for a in yhat])
    
    boxes = list()
    for i in range(len(yhat)):
    # decode the output of the network
        boxes += decode_netout(yhat[i][0], anchors[i], class_threshold, target_h, target_w)
    # correct the sizes of the bounding boxes for the shape of the image
    correct_yolo_boxes(boxes, image_h, image_w, target_h, target_w)
    # suppress non-maximal boxes
    do_nms(boxes, 0.5)
        
    # get the details of the detected objects
    v_boxes, v_labels, v_scores = get_boxes(boxes, labels, class_threshold)
        
    
    if draw_boxes_flag:    
        draw_boxes(photo_filename, v_boxes, v_labels, v_scores)
            
    # for i in range(len(v_boxes)): # for all persons detected by the image
    #     print(v_labels[i], v_scores[i], v_boxes[i].getCoordinates())
            
    return v_boxes, v_scores, v_scores

if __name__ == '__main__':

    # define the expected input shape for the model
    target_w, target_h = 416, 416
    # target_w, target_h = 320, 240

    yolo_model = load_model('model.h5')
    print("model loaded")
    photo_filenames = []
    
    photo_filenames.append('/Users/vcroopana/python_workspace/compVision/cv_proj_py/data/RedoImgs/Img_23.png')
    photo_filenames.append('/Users/vcroopana/python_workspace/compVision/cv_proj_py/data/RedoImgs/Img_21.png')
    photo_filenames.append('/Users/vcroopana/python_workspace/compVision/cv_proj_py/data/RedoImgs/Img_64.png')
    photo_filenames.append('/Users/vcroopana/python_workspace/compVision/cv_proj_py/data/RedoImgs/Img_144.png')

    for photo_filename in photo_filenames:
        
        v_boxes, v_labels, v_scores = getYOLOBBox(yolo_model, photo_filename, target_w, target_h, True, class_threshold=0.6)
        
        for i in range(len(v_boxes)):
        	print(v_labels[i], v_scores[i], v_boxes[i].getCoordinates())
        draw_boxes(photo_filename, v_boxes, v_labels, v_scores)
        print("nboxes:"+ str(len(v_labels)))
    
