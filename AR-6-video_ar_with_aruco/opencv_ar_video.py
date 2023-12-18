# https://pyimagesearch.com/2021/01/11/opencv-video-augmented-reality/?_ga=2.61942508.1361679024.1702885496-1842902230.1698424416
# usage: python opencv_ar_video.py --input jp_trailer_short.mp4
# usage: python opencv_ar_video.py --input jp_trailer_short.mp4 --cache 0
# usage: python opencv_ar_video.py --input jp_trailer_short.mp4 --output results.mp4
from pyimagesearch.augmented_reality import find_and_warp
from imutils.video import VideoStream
from collections import deque # this provides a queue data structure (FIFO)
import argparse
import imutils
import time
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, required=True, help="path to input video file")
ap.add_argument("-c", "--cache", type=int, default=1, help="whether or not to use the cache")
ap.add_argument("-o", "--output", type=str, default="", help="output video stream. default is \"\", in which case, the output is NOT saved")
args = vars(ap.parse_args())

# load ArUCo dictionary
arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
arucoParams = cv2.aruco.DetectorParameters()

# load video that will be warped onto the live video stream
vf = cv2.VideoCapture(args["input"])

# initialize queue to maintain next frame from video stream
# by having at least one frame maintained in the queue, 
# latency in loading the video will be reduced
source_queue = deque(maxlen=128)

# read video capture, get frame and save it in queue
(grabbed, source) = vf.read()
source_queue.appendleft(source)

vs = VideoStream(src=0).start()
vo = None
if args["output"] != "":
    fourcc = cv2.VideoWriter_fourcc(*"MP4V")
    vo = cv2.VideoWriter(args["output"], fourcc, 30.0, (1280, 720))
time.sleep(2.0)
key = cv2.waitKey(1) & 0xFF

# access the frames from the queue
# how the source video is handled:
# if the tags in the camera is detected, then get the oldest source frame
# that was stored in the queue and warp it onto the camera's frame
# if the aruco tags in the camera were not detected, then just store the most recent
# frame into the queue to await warping. That way, the source video is "paused"
# until the aruco tags are missing
# when the source video is done loading, the queue will stop appending items,
# and the video will continue to appear on the camera's frames as long as:
# 1. the queue doesn't run out
# 2. the aruco tags continue to be detected
while len(source_queue) > 0 and key != ord("q"):
    # get frame from live video stream
    frame = vs.read()

    warped_frame = find_and_warp(
        frame,
        source,
        tagIDs=(923, 1001, 241, 1007),
        arucoDict=arucoDict,
        arucoParams=arucoParams,
        useCache=args["cache"] > 0
    )

    if (vo is not None):
        vo.write(warped_frame)

    if warped_frame is not None: # the warp was successful
        frame = warped_frame
        source = source_queue.popleft()

    if len(source_queue) != source_queue.maxlen:
        (grabbed, next_source) = vf.read()

        if grabbed: # i.e., not at the end of the video
            source_queue.append(next_source)
    
    cv2.imshow("frame", frame)
    key = cv2.waitKey(1) & 0xFF

cv2.destroyAllWindows()
vs.stop()
if vo is not None:
    vo.release()