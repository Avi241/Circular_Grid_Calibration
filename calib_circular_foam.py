import cv2
import glob
import numpy as np
 
images_folder = './Thermal/*.jpg'
images_names = sorted(glob.glob(images_folder))
images = []
for imname in images_names:
    im = cv2.imread(imname, 1)
    images.append(im) 

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

########################################Blob Detector##############################################

# Setup SimpleBlobDetector parameters.
blobParams = cv2.SimpleBlobDetector_Params()

# Change thresholds
blobParams.minThreshold = 40
blobParams.maxThreshold = 255

# Filter by Area.
blobParams.filterByArea = True
blobParams.minArea = 64     # minArea may be adjusted to suit for your experiment
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

# Create a detector with the parameters
blobDetector = cv2.SimpleBlobDetector_create(blobParams)

rows = 5
columns = 7
scaling_factor = 1

objp = []
for i in range(columns):
    if i % 2 == 0:
        for j in range(0,rows * 2, 2):
            objp.append([i,j,0])
    else:
        for j in range(1,rows * 2, 2):
            objp.append([i,j,0])
        
objp = np.array(objp,dtype=np.float32)
objp = objp* scaling_factor

# objp = np.zeros((rows*columns,3), np.float32)
# objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
# objp = 72* objp

###################################################################################################

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

for img in images:

    # img = cv2.resize(img,(640,480))
    img = cv2.flip(img,1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    keypoints = blobDetector.detect(gray) # Detect blobs.
    # Draw detected blobs as red circles. This helps cv2.findCirclesGrid() .
    im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    im_with_keypoints_gray = cv2.cvtColor(im_with_keypoints, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findCirclesGrid(im_with_keypoints, (rows,columns), None, flags = cv2.CALIB_CB_ASYMMETRIC_GRID)   # Find the circle grid

    if ret == True:
        objpoints.append(objp)  # Certainly, every loop objp is the same, in 3D.

        # corners = cv2.cornerSubPix(im_with_keypoints_gray, corners, (11,11), (-1,-1), criteria)    # Refines the corner locations.
        imgpoints.append(corners)

        # Draw and display the corners.
        im_with_keypoints = cv2.drawChessboardCorners(img, (rows,columns), corners, ret)

    cv2.imshow("img", im_with_keypoints) # display
    cv2.waitKey(600)

cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print(ret)