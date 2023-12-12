import apriltag
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image with AprilTag")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (7, 7), 0)

# instantiate AprilTags detector
# Essentially, go through each valid family of tags to see which one yields successful detections
# Typically, one would use Tag36h11
# The detection is robust even for rotated images
options = apriltag.DetectorOptions(families="tag36h11") # To see a list of possible families, refer to `python_apriltag_families.webp`
detector = apriltag.Detector(options)
results = detector.detect(blurred)
print(f"{len(results)} number of AprilTags were detected")

for r in results:
    # extract bounding box coordinates:
    (a, b, c, d) = r.corners
    a = (int(a[0]), int(a[1]))
    b = (int(b[0]), int(b[1]))
    c = (int(c[0]), int(c[1]))
    d = (int(d[0]), int(d[1]))

    # draw lines:
    cv2.line(image, a, b, (0, 0, 255), 2)
    cv2.line(image, b, c, (0, 0, 255), 2)
    cv2.line(image, c, d, (0, 0, 255), 2)
    cv2.line(image, d, a, (0, 0, 255), 2)

    # draw center of the bounding box:
    centerX = int( r.center[0] )
    centerY = int( r.center[1] )
    cv2.circle(image, (centerX, centerY), 4, (255, 0, 0), -1)

    tag_family = r.tag_family.decode("utf-8")
    cv2.putText(image, tag_family, (a[0], a[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2 )
cv2.imshow("apriltag", image)
cv2.waitKey(0)