#!/usr/bin/env python

import cv2 as cv
import numpy as np
import sys
import csv

bin_n = 16 # Number of bins

# histogram of gradients
# @param img image
# @returns 64-bit vector with hog
## [hog]
def hog(img):
    gx = cv.Sobel(img, cv.CV_32F, 1, 0)
    gy = cv.Sobel(img, cv.CV_32F, 0, 1)
    mag, ang = cv.cartToPolar(gx, gy)
    bins = np.int32(bin_n*ang/(2*np.pi))    # quantizing binvalues in (0...16)
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)     # hist is a 64 bit vector
    return hist
## [hog]

def color_histogram(img):
    bins_per_color = 128
    chans = cv.split(img)
    colors = ("b", "g", "r")
    features = []
    for (chan, color) in zip(chans, colors):
        hist, bins= np.histogram(chan,bins_per_color,[0,256])
        features.extend(hist)
    return np.array(features).flatten()


def hogcv(img):
    winSize = (20,20)
    blockSize = (10,10)
    blockStride = (5,5)
    cellSize = (10,10)
    nbins = 8
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.5
    gammaCorrection = 1
    nlevels = 32
    signedGradients = True

    h = cv.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma, histogramNormType, L2HysThreshold, gammaCorrection, nlevels, signedGradients)
    return h.compute(img)

def feature_vector(img):
    features = []
#    features.extend(hog(img));
    features.extend(hogcv(img));
    features.extend(color_histogram(img))
    return np.array(features).flatten()

if __name__ == '__main__':

    if len(sys.argv) != 2:
        print ("usage: {0} anot.csv".format(sys.argv[0]))
        print ("  anot.csv consits of \"filename,class\\n\"");
        sys.exit(1)

    anot_file = sys.argv[1];

    cells = []
    responses = []
    classes = []
    filenames = []

    # reads anotation from csv and load images into cells and extract hog
    file_count=0
    with open(anot_file, 'rt') as csvfile:
        anot_reader = csv.reader(csvfile, delimiter=',')
        for row in anot_reader:
            img_filename = row[0]
            anot = row[1]
            img = cv.imread(img_filename)
            if img is None:
                raise Exception("Error: file {} is not image".format(img_filename))

            cells.append(feature_vector(img));
            if not anot in classes:
                classes.append(anot)
            anot_id = classes.index(anot)

            responses.append(anot_id);
            filenames.append(img_filename);

            file_count += 1

    # split into training and testing set
    ratio = 0.8

    train_size = (int)(file_count * ratio)
    print ("train_size={0}; test_size={1}".format(train_size, file_count - train_size))

#    print("data train + anot = :{0} + {1} = {2}".format(train_size, test_size, file_count))

    cells = np.array(cells)

    feature_shape = cells[0].shape[0]

    train_cells = np.array(cells[:train_size])
    test_cells =  np.array(cells[train_size:])

    train_anot=np.array(responses)[:train_size]
    test_anot=np.array(responses)[train_size:]

    test_filenames=np.array(filenames)[train_size:]

    train_data = np.float32(train_cells).reshape(-1,feature_shape )
    test_data = np.float32(test_cells).reshape(-1,feature_shape )

    svm = cv.ml.SVM_create()
#    svm.setKernel(cv.ml.SVM_LINEAR)
    svm.setKernel(cv.ml.SVM_RBF)

    svm.setType(cv.ml.SVM_C_SVC)

    svm.setC(2.67)
    svm.setGamma(5.383)


#    print("c = {0}; gamma = {1}; isTrained = {2}".format(svm.getC(), svm.getGamma(), svm.isTrained()))

    # train_data.shape = (2500,feature_shape), responses.shape: (2500, 1)
#    svm.train(train_data, cv.ml.ROW_SAMPLE, train_anot)
    svm.trainAuto(train_data, cv.ml.ROW_SAMPLE, train_anot)

    print("c = {0}; gamma = {1}; isTrained = {2}".format(svm.getC(), svm.getGamma(), svm.isTrained()))

    svm.save('svm_data.dat')

    result = svm.predict(test_data)[1][:,0]

    mask = result==test_anot

    if 1:
##   print false detections
        print ("<h2>false positive</h2>");
        for f, r, a, m in zip(test_filenames, result, test_anot, mask):
            if not m and a:
                print("<img src=\"file://{}\"/>".format(f))
#                print(r, a, m, f)
        print ("<h2>false negative</h2>");
        for f, r, a, m in zip(test_filenames, result, test_anot, mask):
            if not m and not a:
                print("<img src=\"file://{}\"/>".format(f))

        print ("<h2>true negative</h2>");
        for f, r, a, m in zip(test_filenames, result, test_anot, mask):
            if m and a:
                print("<img src=\"file://{}\"/>".format(f))
#                print(r, a, m, f)

        print ("<h2>true positive </h2>");
        for f, r, a, m in zip(test_filenames, result, test_anot, mask):
            if m and not a:
                print("<img src=\"file://{}\"/>".format(f))
#                print(r, a, m, f)

    correct = np.count_nonzero(mask)
    print("accuracy = {}".format(correct*100.0/result.size))

