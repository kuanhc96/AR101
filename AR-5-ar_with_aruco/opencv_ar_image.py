# https://pyimagesearch.com/2021/01/04/opencv-augmented-reality-ar/?_ga=2.249952902.1816397012.1702485325-1842902230.1698424416
# usage: python opencv_ar_image.py --image input_03.jpg --source alice_and_janet.jpeg
import numpy as np
import argparse
import imutils
import sys
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image with ArUCo tag")
ap.add_argument("-s", "--source", required=True, help="path to source image that will be put on ArUCo image")
args = vars(ap.parse_args())

# load images
image = cv2.imread(args["image"])
# image = imutils.resize(image, width=600)
(imgH, imgW) = image.shape[:2]
source = cv2.imread(args["source"])

# ArUCo detection
arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
arucoParams = cv2.aruco.DetectorParameters()

# Note that the order in which the id's are presented in the `ids` array is arbitrary --
# The detector does not actually understand "Orientation"!
(tags, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)

if (len(tags) != 4):
    print(f"missing coreners! Only {len(tags)} detected")
    sys.exit(0)

ids = ids.flatten()
print("ids", ids)

# Need a way to identify the relative positions (TL, TR, BR, BL) of the detected tags
# As of now, they need to be identified manually, i.e., detect them on an image
# first with the detector, note down which tag belongs to which corner
reference_points = []

# loop over the tags in the following order: TL, TR, BR, BL
for i in (923, 1001, 241, 1007):
    # np.where:
    # inputs:
    # 1 condition: if the elements in the array meet this condition... return input 2, otherwise, return input 3
    # The condition should consist of a broadcastable array meeting some element-wise boolean condition
    # 2 x: if the element in the array meets the condition in input 1 , return x
    # 3 y: if the element in the array does not meet the condition in input 1 , return y
    # if x, y are not provided, this array behaves like: np.asarray(condition).nonzero()
    # np.asarray(condition) will take an array, perform the boolean operation on each element, and return the
    # boolean array as its output. array.nonzero() will return the indices of the non-zero elements of the array,
    # i.e., array.nonzero() will return another ARRAY, even if there is only one element in that array
    j = np.where(ids == i)
    # j will be the index of the current id (i) in the `ids` array
    j = np.squeeze(j) # we are expecting only 1 element to meet the boolean condition above, so we just "squeeze" that single element into a scalar
    tag = np.squeeze(tags[j])
    reference_points.append(tag)

reference_TL_tag = reference_points[0]
reference_TR_tag = reference_points[1]
reference_BR_tag = reference_points[2]
reference_BL_tag = reference_points[3]

destination_point_TL = reference_TL_tag[0] # get the top left corner of the top left tag as the top left destination for the transformation
destination_point_TR = reference_TR_tag[1] # get the top right corner of the top right tag as the top right destination for the transformation
destination_point_BR = reference_BR_tag[2] # get the bottom right corner of the bottom right tag as the bottom right destination for the transformation
destination_point_BL = reference_BL_tag[3] # get the bottom left corner of the bottom left tag as the bottom left destination for the transformation

destination_coordinates = np.array([destination_point_TL, destination_point_TR, destination_point_BR, destination_point_BL])
print(destination_coordinates)

(srcH, srcW) = source.shape[:2]
# get the points of the source image that will be mapped to the points of the destination:
# note that these coordinates are expressed in (x, y)
source_TL = (0, 0)
source_TR = (srcW, 0)
source_BR = (srcW, srcH)
source_BL = (0, srcH)
source_coordinates = np.array([source_TL, source_TR, source_BR, source_BL])
print(source_coordinates)

# Homography: https://learnopencv.com/homography-examples-using-opencv-python-c/
# A homography is a linear transformation matrix that will map a set of source coordinates to a set of destination coordinates
# let's say, the destination coordinates of a cartesian system is: dest_M = [x2, y2, 1]
# let's say, the source coordinates of a cartesian system is: src_M = [x1, y1, 1]
# let's say, the Homography matrix is a 2D, 3x3 matrix:
# H =
#[
# [h00, h01, h02],
# [h10, h11, h12],
# [h20, h21, h22]
# ]
# the Homographic transformation is performed by executing: dest_M = H * src_M
# if the destination space is larger than that of the source, then certain destination coordinates will be left unpopulated, i.e., x2=y2=0
# we are transformting the source coordinates of an image to the destination coordinates of an image on  a cartesian plane
# use cv2.findHomography to calculate the homography matrix
# cv2.findHomography:
# inputs:
# 1 source coordinates: [x1, y1]
# 2 destination coordinates: [x2, y2]
# returns:
# 1 H: the homography matrix that can perform the linear transformation to map [x1, y1] to [x2, y2]
# 2 Mask: not sure what this does (?)
(H, mask)  = cv2.findHomography(source_coordinates, destination_coordinates)
# cv2.warpPerspective:
# performs the Homographic transformation for all points in the source to a destination canvas with size (W, H) using the homography matrix, H
# inputs:
# 1 source: the source image with all source coordinates
# 2 H: the homography matrix
# 3 size: the size of the destination image
warped = cv2.warpPerspective(source, H, (imgW, imgH))
cv2.imshow("warped", warped)
cv2.waitKey(0)

mask = np.ones((imgH, imgW), dtype='float')
# fill a polygon with a certain color
cv2.fillConvexPoly(mask, destination_coordinates.astype("int32"), (0, 0, 0), cv2.LINE_AA)

# mask = mask / 255.0 # turn the mask into a binary image
mask = np.dstack([mask] * 3) # stack the mask such that it shares the same dimensions as GBR images

# warped_in_mask = cv2.multiply(warped.astype(float), mask)
mask_in_image = cv2.multiply(image.astype(float), mask) # essentially, do an element by element multiplication
output = cv2.add(warped.astype(float), mask_in_image) # element by element addition
output = output.astype("uint8") # convert the data type to uint8 so that it can be displayed
# cv2.imshow("warped in mask", warped_in_mask.astype("uint8"))

# make sure that during the final display, the input image (matrix) has to have type uint8
cv2.imshow("mask in image", mask_in_image.astype("uint8"))
cv2.imshow("output", output)
cv2.waitKey(0)

mask = np.ones((imgH, imgW), dtype='uint8') * 255
# fill a polygon with a certain color
cv2.fillConvexPoly(mask, destination_coordinates.astype("int32"), (0, 0, 0), cv2.LINE_AA)


# warped_in_mask = cv2.multiply(warped.astype(float), mask)
mask_in_image = cv2.bitwise_and(image, image, mask=mask)
output = cv2.add(warped, mask_in_image)
output = output.astype("uint8") # convert the data type to uint8 so that it can be displayed
# cv2.imshow("warped in mask", warped_in_mask.astype("uint8"))

# make sure that during the final display, the input image (matrix) has to have type uint8
cv2.imshow("mask in image 2", mask_in_image.astype("uint8"))
cv2.imshow("output 2", output)
cv2.waitKey(0)