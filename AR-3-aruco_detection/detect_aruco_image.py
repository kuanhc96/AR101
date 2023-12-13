import argparse
import imutils
import cv2
import sys

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image containing ArUCo tag")
ap.add_argument("-t", "--type", type-str, default="DICT_ARUCO_ORIGINAL", help="type of ArUCo tag to detect")

args = vars(ap.parse_args())