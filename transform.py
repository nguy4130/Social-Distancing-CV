# import the necessary packages
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
from scipy.spatial import distance as dist
import numpy as np
import cv2

def order_points_flawed(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")

	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	# return the ordered coordinates
	return rect

def order_points(pts):
	# sort the points based on their x-coordinates
	xSorted = pts[np.argsort(pts[:, 0]), :]
	# grab the left-most and right-most points from the sorted
	# x-roodinate points
	leftMost = xSorted[:2, :]
	rightMost = xSorted[2:, :]
	# now, sort the left-most coordinates according to their
	# y-coordinates so we can grab the top-left and bottom-left
	# points, respectively
	leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
	(tl, bl) = leftMost
	# now that we have the top-left coordinate, use it as an
	# anchor to calculate the Euclidean distance between the
	# top-left and right-most points; by the Pythagorean
	# theorem, the point with the largest distance will be
	# our bottom-right point
	D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
	(br, tr) = rightMost[np.argsort(D)[::-1], :]
	# return the coordinates in top-left, top-right,
	# bottom-right, and bottom-left order
	return np.array([tl, tr, br, bl], dtype="float32")

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect

	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))

	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))

	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
# 	dst = np.array([
# 		[0, 0],
# 		[maxWidth - 1, 0],
# 		[maxWidth - 1, maxHeight - 1],
# 		[0, maxHeight - 1]], dtype = "float32")
    
	dst = np.array([
		[tl[0], tl[1]],
		[tl[0]+ maxWidth - 1, tl[1]],
		[tl[0]+ maxWidth - 1, tl[1] + maxHeight - 1],
		[tl[0], tl[1]+ maxHeight - 1]], dtype = "float32")

	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	warped_whole = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]))
    
	patch_orig = image[int(tl[1]):int(bl[1]), int(tl[0]):int(tr[0])]
	if patch_orig.shape[0]==0 or patch_orig.shape[1]==0:
		print("pts:"+ str(pts))
		print("rect:"+ str(rect))
	patch_warped = warped_whole[int(tl[1]):int(bl[1]), int(tl[0]):int(tr[0])]                    
        
	return patch_orig, patch_warped, rect, dst, M, warped, warped_whole

def resizePatches(patch_orig, patch_warped, rect, dst, image, image_warped):

    TARGET_PATCH_SIZE = (80, 80)
    
    patch_warped_resize = cv2.resize(patch_warped, TARGET_PATCH_SIZE)
    x_ratio = patch_warped.shape[0]/TARGET_PATCH_SIZE[0]
    y_ratio = patch_warped.shape[1]/TARGET_PATCH_SIZE[1]
    
    dst_scaled = np.zeros([4,2])
    dst_scaled[0,0] = dst[0,0]/x_ratio
    dst_scaled[1,0] = dst[1,0]/x_ratio
    dst_scaled[2,0] = dst[2,0]/x_ratio
    dst_scaled[3,0] = dst[3,0]/x_ratio
    dst_scaled[0,1] = dst[0,1]/y_ratio
    dst_scaled[1,1] = dst[1,1]/y_ratio
    dst_scaled[2,1] = dst[2,1]/y_ratio
    dst_scaled[3,1] = dst[2,1]/y_ratio
    
    patch_orig_resize = cv2.resize(patch_orig, TARGET_PATCH_SIZE)
    x_ratio = patch_orig.shape[0]/TARGET_PATCH_SIZE[0]
    y_ratio = patch_orig.shape[1]/TARGET_PATCH_SIZE[1]
    
    rect_scaled = np.zeros([4,2])
    rect_scaled[0,0] = rect[0,0]/x_ratio
    rect_scaled[1,0] = rect[1,0]/x_ratio
    rect_scaled[2,0] = rect[2,0]/x_ratio
    rect_scaled[3,0] = rect[3,0]/x_ratio
    rect_scaled[0,1] = rect[0,1]/y_ratio
    rect_scaled[1,1] = rect[1,1]/y_ratio
    rect_scaled[2,1] = rect[2,1]/y_ratio
    rect_scaled[3,1] = rect[3,1]/y_ratio
    
    return patch_orig_resize, patch_warped_resize, rect_scaled, dst_scaled

def generate_homography_example(patch_orig, patch_warped, corners_square, perturbed_corners_square):
    # Steps 1-8:
    # Step 9: Stack the patches together depth-wise
    PATCH_SIZE = 20
    example = np.zeros((PATCH_SIZE, PATCH_SIZE, 2))
    # print(example.shape)
    example[:, :, 0] = patch_orig
    example[:, :, 1] = patch_warped

    corners_square = np.array(corners_square)
    perturbed_corners_square = np.array(perturbed_corners_square)

    # Step 10: calculate the difference between the corners of the perturbed patch and the original patch
    # This is the target for the above training example
    prediction = perturbed_corners_square - corners_square
    # print(prediction)

    return example, prediction.flatten()