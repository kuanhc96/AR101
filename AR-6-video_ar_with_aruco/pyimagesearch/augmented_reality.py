import numpy as np
import cv2

# cached reference points used in place of real-time reference points in the event
# the ArUCo markers are not detected
# Without this cache, videos will appear to be flickering when reference points
# are not detected
CACHED_REF_PTS = None

# inputs:
# 1 frame: the input frame from a video stream
# 2 source: the source image that will be warped onto the video stream
# 3 tagIDs: the ID's of the ArUCo tags that need to be detected
# 4 arucoDict: OpenCV's ArUCo tag dictionary
# 5 arucoParams: the ArUCo marker detector parameters
# 6 useCache: boolean -- whether or not to use the cache. True by default
def find_and_warp(frame, source, tagIDs, arucoDict, arucoParams, useCache=True):
    global CACHED_REF_PTS
    # get width of frame and source image
    (frameH, frameW) = frame.shape[:2]
    (sourceH, sourceW) = source.shape[:2]

    # detect ArUCo tags in the input frame:
    (tags, ids, rejected) = cv2.aruco.detectMarkers(frame, arucoDict, parameters=arucoParams)

    ids = np.array([]) if len(tags) != 4 else ids.flatten()

    # initialize list of reference points:
    reference_points = []

    # loop over the ID's of the ArUCo markers: TL, TR, BR, BL
    for i in tagIDs:
        # index of the current id:
        j = np.squeeze(np.where(ids == i))

        if j.size > 0:
            tag = np.squeeze(tags[j])
            reference_points.append(tag)
    
    # at least one tag was not found in the frame
    # consider using cached frames
    if len(reference_points) < 4:
        if useCache and CACHED_REF_PTS is not None:
            reference_points = CACHED_REF_PTS
        else:
            return None

    # record the current reference points in case we need them in the future
    if useCache:
        CACHED_REF_PTS = reference_points
    
    # get destination points
    destination_TL = reference_points[0]
    destination_TR = reference_points[1]
    destination_BR = reference_points[2]
    destination_BL = reference_points[3]
    destination_points = [destination_TL[0], destination_TR[1], destination_BR[2], destination_BL[3]]
    destination_points = np.array(destination_points)

    # warp the source image onto the frame
    source_points = np.array([(0, 0), (sourceW, 0), (sourceW, sourceH), (0, sourceH)])
    (H, _) = cv2.findHomography(source_points, destination_points)
    warped = cv2.warpPerspective(source, H, (frameW, frameH))
    warped = warped.astype("uint8")

    # make a mask
    mask = np.ones((frameH, frameW), dtype="uint8") * 255
    mask = cv2.fillConvexPoly(mask, destination_points.astype("int32"), (0, 0, 0), cv2.LINE_AA)
    # make the white area slightly smaller:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.erode(mask, kernel, iterations=2)
    # apply the mask
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
    masked_frame = cv2.add(masked_frame, warped)
    return masked_frame
