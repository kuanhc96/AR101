import numpy as np
import argparse
import cv2
import sys

ap = argparse.ArgumentParser()
# output path to the ArUCo tage that will be generated
ap.add_argument("-o", "--output", required=True, help="path to output image containing ArUCo tag")
# the unique ID of the ArUCo tag generated. The ID must be a valid ID in the ArUCo dictionary used for generation
ap.add_argument("-i", "--id", type=int, required=True, help="ID of ArUCo tag to generate")
# the name of the ArUCo dictionary that will be used to generate the tag
ap.add_argument("-t", "--type", type=str, default="DICT_ARUCO_ORIGINAL", help="type of ArUCo tag to generate")
args = vars(ap.parse_args())

ARUCO_DICT = {
	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL, # Default dict
	"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
	"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
	"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
	"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

# verify that the supplied ArUCo tag exists:
if ARUCO_DICT.get(args["type"], None) is None:
    print(f"ArUCo tage {args['type']} is not supported")
    sys.exit(0)

arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[args["type"]])
print(f"generating ArUCo tag type {args['type']}, with ID {args['id']}")
tag = np.zeros((300, 300, 1), dtype="uint8")
# cv2.aruco.drawMarker:
# Inputs:
# 1 dictionary: dictionary of markers indicating the type of markers
# 2 id: ID of the marker that will be returned. Has to be a valid ID in the specified dictionary
# 3 sidePixels: The size of the image in pixels
# 4 image: the array where the output marker (image) will be stored
# 5 border: the width of the padding (border) around the marker
cv2.aruco.generateImageMarker(arucoDict, args["id"], 300, tag, 1)
cv2.imwrite(args["output"], tag)
cv2.imshow("ArUCo tag", tag)
cv2.waitKey(0)