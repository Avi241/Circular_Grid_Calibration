import cv2 as cv
import glob
import numpy as np
 
def calibrate_camera(images_folder):
    images_names = sorted(glob.glob(images_folder))
    images = []
    for imname in images_names:
        im = cv.imread(imname, 1)
        images.append(im)
 
    #criteria used by checkerboard pattern detector.
    #Change this if the code can't find the checkerboard
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
    # Setup SimpleBlobDetector parameters.
    blobParams = cv.SimpleBlobDetector_Params()

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
    blobDetector = cv.SimpleBlobDetector_create(blobParams)

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

        # img = cv.resize(img,(640,480))
        img = cv.flip(img,1)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        keypoints = blobDetector.detect(gray) # Detect blobs.
        # Draw detected blobs as red circles. This helps cv.findCirclesGrid() .
        im_with_keypoints = cv.drawKeypoints(img, keypoints, np.array([]), (0,255,0), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        im_with_keypoints_gray = cv.cvtColor(im_with_keypoints, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findCirclesGrid(im_with_keypoints, (rows,columns), None, flags = cv.CALIB_CB_ASYMMETRIC_GRID)   # Find the circle grid

        if ret == True:
            objpoints.append(objp)  # Certainly, every loop objp is the same, in 3D.

            # corners = cv.cornerSubPix(im_with_keypoints_gray, corners, (11,11), (-1,-1), criteria)    # Refines the corner locations.
            imgpoints.append(corners)

            # Draw and display the corners.
            im_with_keypoints = cv.drawChessboardCorners(img, (rows,columns), corners, ret)

        cv.imshow("img", im_with_keypoints) # display
        cv.waitKey(600)

    cv.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return mtx, dist
 
mtx1, dist1 = calibrate_camera(images_folder = './RGB/*')
mtx2, dist2 = calibrate_camera(images_folder = './Thermal/*')

def stereo_calibrate(mtx1, dist1, mtx2, dist2, frames_folder1,frame_folder2):
    #read the synched frames
    c1_images_names= sorted(glob.glob(frames_folder1))
    c2_images_names = sorted(glob.glob(frame_folder2))
 
    c1_images = []
    c2_images = []

    for im1, im2 in zip(c1_images_names, c2_images_names):
        _im = cv.imread(im1, 1)
        c1_images.append(_im)
 
        _im = cv.imread(im2, 1)
        c2_images.append(_im)
 
    #change this if stereo calibration not good.
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
 
    blobParams = cv.SimpleBlobDetector_Params()

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
    blobDetector = cv.SimpleBlobDetector_create(blobParams)

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
    imgpoints_left = [] # 2d points in image plane.
    imgpoints_right = []

    for frame1, frame2 in zip(c1_images, c2_images):

        # img = cv.resize(img,(640,480))
        gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

        keypoints1 = blobDetector.detect(gray1) # Detect blobs.
        keypoints2 = blobDetector.detect(gray2) # Detect blobs.

        # Draw detected blobs as red circles. This helps cv.findCirclesGrid() .
        im_with_keypoints1 = cv.drawKeypoints(frame1, keypoints1, np.array([]), (0,255,0), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        im_with_keypoints_gray1 = cv.cvtColor(im_with_keypoints1, cv.COLOR_BGR2GRAY)
        ret1, corners1 = cv.findCirclesGrid(im_with_keypoints1, (rows,columns), None, flags = cv.CALIB_CB_ASYMMETRIC_GRID)   # Find the circle grid

        im_with_keypoints2 = cv.drawKeypoints(frame2, keypoints2, np.array([]), (0,255,0), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        im_with_keypoints_gray2 = cv.cvtColor(im_with_keypoints2, cv.COLOR_BGR2GRAY)
        ret2, corners2 = cv.findCirclesGrid(im_with_keypoints2, (rows,columns), None, flags = cv.CALIB_CB_ASYMMETRIC_GRID)   # Find the circle grid


        if ret1 == True and ret2 == True:
            objpoints.append(objp)  # Certainly, every loop objp is the same, in 3D.

            # corners = cv.cornerSubPix(im_with_keypoints_gray, corners, (11,11), (-1,-1), criteria)    # Refines the corner locations.
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)

            # Draw and display the corners.
            im_with_keypoints1 = cv.drawChessboardCorners(frame1, (rows,columns), corners1, ret1)
            im_with_keypoints2 = cv.drawChessboardCorners(frame2, (rows,columns), corners2, ret2)

        cv.imshow("img", im_with_keypoints1) # display
        cv.imshow("img2", im_with_keypoints2) # display

        cv.waitKey(100)
 
    stereocalibration_flags = cv.CALIB_FIX_INTRINSIC
    ret, CM1, dist1, CM2, dist2, R, T, E, F = cv.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right, mtx1, dist1,
                                                                 mtx2, dist2, (640, 480), criteria = criteria, flags = stereocalibration_flags)

    print(ret)
    return R, T
 
R, t = stereo_calibrate(mtx1, dist1, mtx2, dist2, './RGB/*','./Thermal/*')
print(R,t)