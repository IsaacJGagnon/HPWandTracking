from collections import deque
import cv2
import time
import numpy as np

TrainerImageCols = 64
TrainerImageRows = 64

winSize = (64, 64)
blockSize = (32, 32)
blockStride = (16, 16)
cellSize = (16, 16)
nbins = 9
derivAper = 1
winSigma = -1
histogramNormType = 0
L2HysThresh = 0.2
gammaCor = 0
nlevels = 64

hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins,
                        derivAper, winSigma, histogramNormType, L2HysThresh,
                        gammaCor, nlevels)

img = cv2.imread('Training3.png', cv2.IMREAD_GRAYSCALE)

ImgCount = 0

trainingCells = deque()
trainHOG = deque()

num_rows, num_cols = img.shape

i = 0
while i < num_rows:
    j = 0
    while j < num_cols:
        digitImg = img[i:i+TrainerImageRows, j:j+TrainerImageCols]
        # cv2.imshow("IMG", digitImg)
        # cv2.waitKey(1)
        # time.sleep(2)
        trainingCells.append(digitImg)
        ImgCount += 1
        j += TrainerImageCols
    i += TrainerImageRows

for cell in trainingCells:
    descriptors = hog.compute(cell)
    trainHOG.append(descriptors)

print(len(trainHOG))
desc_size = len(trainHOG[0])
print(desc_size)

labels = np.empty((80, 1), dtype=np.int32)
labels[0:20] = 1
labels[20:40] = 2
labels[40:60] = 3
labels[60:80] = 4

svm = cv2.ml.SVM_create()
svm.setGamma(0.50625)
svm.setC(12.5)
svm.setKernel(cv2.ml.SVM_RBF)
svm.setType(cv2.ml.SVM_C_SVC)
svm.trainAuto(np.asarray(trainHOG), cv2.ml.ROW_SAMPLE, labels)
svm.save("Model3.yml")
