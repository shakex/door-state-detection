import cv2
import numpy as np
from pylab import *

__all__ = ['rgb2bgr', 'bgr2rgb', 'normalize', 'kpt2np',
           'match2np', 'computearea', 'computercenter']


def rgb2bgr(img):
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]
    img_ = np.empty_like(img)
    img_[:, :, 0] = b
    img_[:, :, 1] = g
    img_[:, :, 2] = r


# opecv imread(bgr) -> skimage.io(rgb)
def bgr2rgb(img):
    b, g, r = cv2.split(img)
    img_ = img[:, :, ::-1]
    return img_


#
def normalize(array):
    min, max = array.min(), array.max()
    array = (array - min) / (max - min)
    return array


# convert opencv datatype <keypoint> to <np.array>
def kpt2np(keypoint):
    keypoint_np = np.zeros((len(keypoint), 2))
    for i in range(len(keypoint)):
        keypoint_np[i][0] = keypoint[i].pt[1]
        keypoint_np[i][1] = keypoint[i].pt[0]
    return keypoint_np


# convert opencv datatype <DMatch> to <np.array>
def match2np(matches):
    matches_np = np.zeros((len(matches), 2), dtype=np.int64)
    for i in range(len(matches)):
        matches_np[i][0] = matches[i].queryIdx
        matches_np[i][1] = matches[i].trainIdx
    return matches_np


def computearea(dst):
    xa = dst[0, 0, 0]
    ya = dst[0, 0, 1]
    xb = dst[1, 0, 0]
    yb = dst[1, 0, 1]
    xc = dst[2, 0, 0]
    yc = dst[2, 0, 1]
    xd = dst[3, 0, 0]
    yd = dst[3, 0, 1]
    cd = sqrt((xc - xd) ** 2 + (yc - yd) ** 2)
    bc = sqrt((xb - xc) ** 2 + (yb - yc) ** 2)
    A_1 = yd - yc
    B_1 = xd - xc
    C_1 = yc * (xd - xc) - xc * (yd - yc)
    hcd = abs((A_1 * xa + B_1 * ya + C_1) / sqrt(A_1 ** 2 + B_1 ** 2))

    A_2 = yb - yc
    B_2 = xb - xc
    C_2 = yc * (xb - xc) - xc * (yb - yc)
    hbc = abs((A_2 * xa + B_2 * ya + C_2) / sqrt(A_2 ** 2 + B_2 ** 2))

    area = (cd * hcd + bc * hbc) / 2

    return area


# ([(x3-x1)(x4-x2)(y2-y1)+x1(y3-y1)(x4-x2)-x2(y4-y2)(x3-x1)]/[(y3-y1)(x4-x2)-(y4-y2)(x3-x1)],(y3-y1)[(x4-x2)(y2-y1)+(x1-x2)(y4-y2)]/[(y3-y1)(x4-x2)-(y4-y2)(x3-x1)]+y1)

def computercenter(dst):
    xa = dst[0, 0, 0]
    ya = dst[0, 0, 1]
    xb = dst[1, 0, 0]
    yb = dst[1, 0, 1]
    xc = dst[2, 0, 0]
    yc = dst[2, 0, 1]
    xd = dst[3, 0, 0]
    yd = dst[3, 0, 1]
    x_center = ((xc - xa) * (xd - xb) * (yb - ya) + xa * (yc - ya) * (xd - xb) -
                xb * (yd - yb) * (xc - xa)) / ((yc - ya) * (xd - xb) - (yd - yb) * (xc - xa))
    y_center = (yc - ya) * ((xd - xb) * (yb - ya) + (xa - xb) * (yd - yb)
                            ) / ((yc - ya) * (xd - xb) - (yd - yb) * (xc - xa)) + ya
    return x_center, y_center
