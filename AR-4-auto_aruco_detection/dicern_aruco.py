import cv2
import argparse
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

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True, help="path to input image containing tags")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])

for dict_name, dict_code in ARUCO_DICT.items():
    arucoDict = cv2.aruco.getPredefinedDictionary(dict_code)
    params = cv2.aruco.DetectorParameters()
    (corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict, parameters=params)

    if ids is not None and len(ids) > 0:
        ids = ids.flatten()
        print(f"{len(ids)} tag(s) detected for dictionary type {dict_name}")
        image_copy = image.copy()
        
        for ( corner, id ) in zip( corners, ids ):
            top_left = corner[0][0]
            top_right = corner[0][1]
            bottom_right = corner[0][2]
            bottom_left = corner[0][3]

            top_left = (int(top_left[0]), int(top_left[1]))
            top_right = (int(top_right[0]), int(top_right[1]))
            bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
            bottom_left = (int(bottom_left[0]), int(bottom_left[1]))
            center_x = int((top_left[0] + bottom_right[0]) // 2)
            center_y = int((top_left[1] + bottom_right[1]) // 2)
            
            cv2.line(image_copy, top_left, top_right, (0, 0, 255), 3)
            cv2.line(image_copy, top_right, bottom_right, (0, 0, 255), 3)
            cv2.line(image_copy, bottom_right, bottom_left, (0, 0, 255), 3)
            cv2.line(image_copy, bottom_left, top_left, (0, 0, 255), 3)
            cv2.circle(image_copy, (center_x, center_y), 3, (128, 128, 0), -1)
            cv2.putText(image_copy, str( id ), (top_left[0] + 10, top_left[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        cv2.imshow(f"Detection results for {dict_name}", image_copy)
        
cv2.waitKey(0)