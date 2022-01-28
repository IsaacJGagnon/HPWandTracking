# import the necessary packages
from collections import deque
import math
import serial
import cv2.cv2
import numpy as np  # Provides Numpy related functions
import time
import cv2  # OpenCV library

# Initialize the camera
camera = cv2.VideoCapture(0)

width = 0
height = 0

if camera.isOpened():
    width = camera.get(3)  # get frame width
    height = camera.get(4)  # get frame height

# BlobDetector Params
params = cv2.SimpleBlobDetector_Params()

params.minThreshold = 150
params.maxThreshold = 255

params.filterByColor = True
params.blobColor = 255

params.filterByArea = True
params.minArea = 0.05
params.maxArea = 300

params.filterByCircularity = True
params.minCircularity = 0.5
params.maxCircularity = 1

params.filterByConvexity = True
params.minConvexity = 0.5

params.filterByInertia = False

# Set up the detector with.
detector = cv2.SimpleBlobDetector_create(params)

# Set up Background Subtractor
mog2 = cv2.createBackgroundSubtractorMOG2()

# Set structuring elements
se1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
se2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

# Wand trace variables
tracePts = deque()
lastKPTime = time.time()
max_trace_speed = 450
deque_buffer_size = 40
trace_thickness = 4
wandMoveTracingFrame = np.zeros((int(height), int(width)))
topLeft = [0, 0]
botRight = [1, 1]

# Bluetooth Serial COM
port = 'COM7'
s = serial.Serial(port)

fileNum = 0


def call_spell(spell_num):
    if spell_num == 1:
        print("LOCOMOTOR!!!")
        s.write(b'1')
    elif spell_num == 2:
        print("ARRESTO MOMENTUM!!!")
        s.write(b'2')
    elif spell_num == 3:
        print("SILENCIO!!!")
        s.write(b'3')
    elif spell_num == 4:
        print("FLIPENDO!!!")
        s.write(b'4')


def check_spell(img):
    win_size = (64, 64)
    block_size = (32, 32)
    block_stride = (16, 16)
    cell_size = (16, 16)
    n_bins = 9
    deriv_aper = 1
    win_sigma = -1
    histogram_norm_type = 0
    l2_hys_thresh = 0.2
    gamma_cor = 0
    n_levels = 64

    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, n_bins,
                            deriv_aper, win_sigma, histogram_norm_type, l2_hys_thresh,
                            gamma_cor, n_levels)

    desc = hog.compute(img)

    svm = cv2.ml.SVM_load("Model3.yml")

    return svm.predict(np.asarray([desc]))


# Gets wand trace and checks for spell validity
def get_wand_trace(kps):
    global fileNum
    global topLeft
    global botRight
    global lastKPTime
    global wandMoveTracingFrame

    largest_kp = None
    current_kp_time = time.time()
    elapsed = current_kp_time - lastKPTime

    # Check if there are any kps in the input
    if len(kps) > 0:
        # Find the largest keypoint
        for kp in kps:
            if largest_kp is None:
                largest_kp = kp
            elif kp.size > largest_kp.size:
                largest_kp = kp

        # Check if we have any valid trace points yet
        if len(tracePts) != 0:
            lastKPTime = current_kp_time

            pt1 = (int(tracePts[len(tracePts) - 1].pt[0]), int(tracePts[len(tracePts) - 1].pt[1]))
            pt2 = (int(largest_kp.pt[0]), int(largest_kp.pt[1]))

            if math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1]) / elapsed < max_trace_speed:
                if len(tracePts) >= deque_buffer_size:
                    tracePts.popleft()

                tracePts.append(largest_kp)
                cv2.line(wandMoveTracingFrame, pt1, pt2, 255, 4)

                if pt2[0] > botRight[0]:
                    botRight[0] = pt2[0]
                elif pt2[0] < topLeft[0]:
                    topLeft[0] = pt2[0]

                if pt2[1] > botRight[1]:
                    botRight[1] = pt2[1]
                elif pt2[1] < topLeft[1]:
                    topLeft[1] = pt2[1]
        else:
            # If there arent any valid tracepoints yet, append the first tracepoint
            lastKPTime = current_kp_time
            tracePts.append(largest_kp)
            topLeft = [int(largest_kp.pt[0]), int(largest_kp.pt[1])]
            botRight = [int(largest_kp.pt[0])+1, int(largest_kp.pt[1])+1]

    # If we keypoints is empty and we haven't received a keypoint in 5 seconds
    # time to check for spell validity
    elif elapsed > 5.0:
        # Make sure were not outside of the frame
        topLeft[0] = 0 if topLeft[0]-4 < 0 else topLeft[0]-4
        topLeft[1] = 0 if topLeft[1]-4 < 0 else topLeft[1]-4
        botRight[0] = 640 if botRight[0]+4 > 640 else botRight[0]+4
        botRight[1] = 480 if botRight[1]+4 > 480 else botRight[1]+4

        # Get the cropped image then resize it based on height and width
        cropped_img = wandMoveTracingFrame[int(topLeft[1]):int(botRight[1]),
                                          int(topLeft[0]):int(botRight[0])]
        h, w = cropped_img.shape

        if h > w:
            re_h = 64
            re_w = int(w*64/h)
        else:
            re_w = 64
            re_h = int(h*64/w)

        resized = cv2.resize(cropped_img, (re_w, re_h))
        final = np.zeros((64, 64))

        r = 0
        while r < re_h:
            c = 0
            while c < re_w:
                final[r][c] += resized[r][c]
                c += 1
            r += 1

        cv2.imshow("CAPTURED IMAGE", final)
        cv2.waitKey(1)
        if len(tracePts) >= 30:
            prediction = check_spell(final.astype(np.uint8))
            call_spell(int(prediction[1][0][0]))

            # u_in = input("Would you like to save this image?")
            # if u_in == 'y' or u_in == 'Y':
            #     file_name = "silencioImage" + str(fileNum) + ".png"
            #     cv2.imwrite(file_name, final)
            #     fileNum += 1

        # After checking for spell validity, clear the wandMoveTracingFrame to prep for new input
        wandMoveTracingFrame = np.zeros((int(height), int(width)))
        tracePts.clear()
        topLeft = [0, 0]
        botRight = [1, 1]
        lastKPTime = time.time()


# Capture frames continuously from the camera
while True:
    # Grab the raw NumPy array representing the image
    ret, image = camera.read()

    # convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #cv2.imshow("GRAY", gray_image)

    # convert to black and white
    (thresh, bawImage) = cv2.threshold(gray_image, 250, 255, cv2.THRESH_BINARY)

    # apply background elimination
    bg_elim_image = mog2.apply(bawImage)

    # Eliminate Spurious islands
    isl_elim1 = cv2.morphologyEx(bg_elim_image, cv2.MORPH_CLOSE, se1)
    isl_elim2 = cv2.morphologyEx(isl_elim1, cv2.MORPH_OPEN, se2)

    # Detect blobs.
    keypoints = detector.detect(isl_elim2)

    get_wand_trace(keypoints)

    cv2.imshow("TRACE", wandMoveTracingFrame)

    # ensures the size of the circle corresponds to the size of blob
    # im_with_keypoints = cv2.drawKeypoints(grayImage, keypoints, np.array([]), (0, 0, 255),
    #                                      cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Display frame with keypoints using OpenCV
    # cv2.imshow("IM_WITH_KEYPOINTS", im_with_keypoints)

    # Wait for keyPress for 1 millisecond
    key = cv2.waitKey(1) & 0xFF

    # If the `q` key was pressed, break from the loop
    if key == ord('q'):
        break
