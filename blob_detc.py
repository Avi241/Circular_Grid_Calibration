# Standard imports
import cv2
import numpy as np
import glob

images_folder = './thermal_sync/*.jpg'
images_names = sorted(glob.glob(images_folder))


blobParams = cv2.SimpleBlobDetector_Params()

# Change thresholds
blobParams.minThreshold = 40
blobParams.maxThreshold = 255

# Filter by Area.
blobParams.filterByArea = True
blobParams.minArea = 50     # minArea may be adjusted to suit for your experiment
blobParams.maxArea = 1500   # maxArea may be adjusted to suit for your experiment

blobParams.filterByColor = True
blobParams.blobColor = 0

# Filter by Circularity
blobParams.filterByCircularity = True
blobParams.minCircularity = 0.1

# Filter by Convexity
blobParams.filterByConvexity = True
blobParams.minConvexity = 0.87

# Filter by Inertia
blobParams.filterByInertia = True
blobParams.minInertiaRatio = 0.01

# Set up the detector with default parameters.
detector = cv2.SimpleBlobDetector_create(blobParams)


for imname in images_names:
    im = cv2.imread(imname, cv2.IMREAD_GRAYSCALE)
    
    # # Detect blobs.
    im = cv2.resize(im,(640,480))

    keypoints = detector.detect(im)
    print(len(keypoints))
    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Show keypoints
    cv2.imshow("Keypoints", im_with_keypoints)
    cv2.waitKey(0)

cv2.destroyAllWindows()