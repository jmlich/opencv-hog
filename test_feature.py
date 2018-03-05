#!/usr/bin/env python

import matplotlib.pyplot as plt
import hogsvm
import cv2 as cv
import numpy as np

bin_n = 16 # Number of bins


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


if __name__ == '__main__':

    img_filename = "/var/www/html/fire/labelme_samples/000366_37.jpg"
    img = cv.imread(img_filename)
    if img is None:
        raise Exception("Error: file {0} is not image".format(img_filename))

#    print(img)
    fea = hogsvm.feature_vector(img)
#    fea = hog(img)
    print(fea)
#    print(fea.shape)

#    fea2 = color_hist(img)
#    print(fea2)
#    print(fea2.shape)
