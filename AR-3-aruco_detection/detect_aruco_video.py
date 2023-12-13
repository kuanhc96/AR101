from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2
import sys

ap = argparse.ArgumentParser()
ap.add_argument("-t", "--type", type=str, default="DICT_ARUCO_ORIGINAL", help="type of ArUCo tag to detect")
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
	"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
	"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
	"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
	"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
	"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

if (ARUCO_DICT.get(args["type"], None) is None):
    print(f"ArUCo tag {args['type']} is invalid")
    sys.exit(0)

print(f"detecting {args['type']} type ArUCo tags")
arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT.get(args["type"]))

# Get the ArUCo parameters used for detection
# Unless there is a good reason, using the default parameters are generally sufficient to get good results
arucoParams = cv2.aruco.DetectorParameters()

vs = VideoStream(src=0).start()
time.sleep(2.0)

key = cv2.waitKey(1) & 0xFF
while key != ord("q"):
    frame = vs.read()
    frame = imutils.resize(frame, width=1000)

    (corners, ids, rejected) = cv2.aruco.detectMarkers(frame, arucoDict, parameters=arucoParams)

    if ids is not None and len(ids > 0):
        ids = ids.flatten()

        # loop over the potential markers detected
        for (corner, id) in zip(corners, ids):
            top_left = corner[0][0]
            top_right = corner[0][1]
            bottom_right = corner[0][2]
            bottom_left = corner[0][3]
            center_x = int((top_right[0] + bottom_left[0]) // 2)
            center_y = int((top_right[1] + bottom_left[1]) // 2)
            top_left =     (int(top_left[0]), int(top_left[1]))
            top_right =    (int(top_right[0]), int(top_right[1]))
            bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
            bottom_left =  (int(bottom_left[0]), int(bottom_left[1]))
            cv2.line(frame, top_left, top_right, (0, 0, 255), 3)
            cv2.line(frame, top_right, bottom_right, (0, 0, 255), 3)
            cv2.line(frame, bottom_right, bottom_left, (0, 0, 255), 3)
            cv2.line(frame, top_left, bottom_left, (0, 0, 255), 3)
            cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)
            cv2.putText(frame, str(id), (top_left[0] + 3, top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    cv2.imshow("VideoStream", frame)
    key = cv2.waitKey(1) & 0xFF

cv2.destroyAllWindows()
vs.stop()