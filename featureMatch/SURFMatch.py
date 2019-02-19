import cv2
import numpy as np
import skimage.draw as dr
import matplotlib.pyplot as plt
import _functions as fun
from skimage.feature import *
from pylab import *

class SURFMatch(object):
    """ match local feature points with SURF """

    def init(self, pattern, scene, hessianThreshold = 5000, ratio = 0.85, min_match_count = 4):
        """
        Set surf match parameters.

        Inputs:
        - pattern: pattern image
        - scene: scene image
        - hessianThreshold: number of detected keypoints
        - ratio: set ratio to find good mathches
        - min_match_count
        """
        self.pattern = pattern
        self.scene = scene
        self.hessianThreshold = hessianThreshold
        self.ratio = ratio
        self.min_match_count = min_match_count

        #
        self.kpts1 = None
        self.kpts2 = None
        self.matches = None
        self.dst = np.zeros((4, 1, 2), float32)

    def surfCompute(self):
        """
        detect&descript SURF features and find pattern from the scene image

        Values:

        - self.kpts1: numpy array of keypoints from pattern image
        (shape of (num_of_kpts1, 2))
        - self.kpts2: numpy array of keypoints from scene image
        (shape of (num_of_kpts2, 2))
        - self.matches: numpy array of matches between pattern image and scene image
        (shape of (index_of_matched_kpts, 2))
        - self.dst: location of four corners in scene image
        (shape of (4, 1, 2))
        """

        # Create surf object
        surf = cv2.xfeatures2d.SURF_create(self.hessianThreshold)

        # Create flann matcher
        matcher = cv2.FlannBasedMatcher(dict(algorithm = 1, trees = 5), {})

        # Detect keypoints and compute keypointer descriptors
        kpts1, descs1 = surf.detectAndCompute(self.pattern, None)
        kpts2, descs2 = surf.detectAndCompute(self.scene, None)

        # KnnMatch to get top2
        matches = matcher.knnMatch(descs1, descs2, 2)
        matches = sorted(matches, key = lambda x:x[0].distance)

        # Ratio test to get good matches
        good = [m1 for (m1, m2) in matches if m1.distance < self.ratio * m2.distance]

        # Convert to numpy array
        self.kpts1 = fun.kpt2np(kpts1)
        self.kpts2 = fun.kpt2np(kpts2)
        self.matches = fun.match2np(good)

        # Find homography matrix
        if len(self.matches) > self.min_match_count:
            src_pts = np.float32([kpts1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kpts2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            # find homography matrix in cv2.RANSAC using good match points
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if M is None:
                self.dst = np.zeros((4, 1, 2), float32)
                return
            h, w = self.pattern.shape[:2]
            pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
            self.dst = cv2.perspectiveTransform(pts, M)
        # else:
            # print( "Not enough matches are found - {}/{}".format(len(self.matches), self.min_match_count))


    def computeArea(self):
        """
        Compute the areas.

        Output:
        - area
        """
        if self.dst[0, 0, 0] == 0:
            return -1

        xa = self.dst[0, 0, 0]
        ya = self.dst[0, 0, 1]
        xb = self.dst[1, 0, 0]
        yb = self.dst[1, 0, 1]
        xc = self.dst[2, 0, 0]
        yc = self.dst[2, 0, 1]
        xd = self.dst[3, 0, 0]
        yd = self.dst[3, 0, 1]
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


    def computeCenter(self):
        """
        Compute the center point of the detected object.

        Output:
        - cneter: numpy array of the center point.
        (shape of (2,))
        """

        if self.dst[0, 0, 0] == 0:
            return [-1, -1]

        xa = self.dst[0, 0, 0]
        ya = self.dst[0, 0, 1]
        xb = self.dst[1, 0, 0]
        yb = self.dst[1, 0, 1]
        xc = self.dst[2, 0, 0]
        yc = self.dst[2, 0, 1]
        xd = self.dst[3, 0, 0]
        yd = self.dst[3, 0, 1]
        x_center = ((xc - xa) * (xd - xb) * (yb - ya) + xa * (yc - ya) * (xd - xb) -
                    xb * (yd - yb) * (xc - xa)) / ((yc - ya) * (xd - xb) - (yd - yb) * (xc - xa))
        y_center = (yc - ya) * ((xd - xb) * (yb - ya) + (xa - xb) * (yd - yb)
                    ) / ((yc - ya) * (xd - xb) - (yd - yb) * (xc - xa)) + ya

        center = np.zeros(2)
        center[0] = x_center
        center[1] = y_center

        return center


    def plotMatch(self, **kwargs):
        """
        Plot matches between pattern image and scene image
        """

        # set some default parameters
        kwargs.setdefault('keypoints_color', 'k')
        kwargs.setdefault('matches_color', 'g')
        kwargs.setdefault('only_matches', False)

        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        plot_matches(ax, fun.bgr2rgb(self.pattern), fun.bgr2rgb(self.scene), self.kpts1, self.kpts2, self.matches, **kwargs)
        plt.title('SURF: Keypoints Match (# of matched: %d)' % self.matches.shape[0])
        ax.axis('off')

        plt.subplots_adjust(left=0.05,  bottom=0.05,
                            right=0.95, top=0.9,
                            hspace=0.1, wspace=0.05)


    def plotFind(self, **kwargs):
        """
        Locate the pattern from the scene image
        """

        # set some default parameters
        # kwargs.setdefault('edgecolor', 'k')
        # kwargs.setdefault('facecolor', 'g')
        # kwargs.setdefault('linewidth', 2)
        # kwargs.setdefault('alpha', 0.5)

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        # plot scene image
        # ax.imshow(fun.bgr2rgb(self.scene), cmap=plt.cm.gray)
        
        # draw rectangle
        # width = self.dst[2, 0, 0] - self.dst[0, 0, 0]
        # height = self.dst[2, 0, 1] - self.dst[0, 0, 1]
        # rect = plt.Rectangle((self.dst[0, 0, 0], self.dst[0, 0, 1]), width, height, **kwargs)
        # ax.add_patch(rect)

        # draw detected object
        img = fun.bgr2rgb(self.scene)
        Y = np.array([self.dst[0, 0, 1], self.dst[1, 0, 1], self.dst[2, 0, 1], self.dst[3, 0, 1]])
        X = np.array([self.dst[0, 0, 0], self.dst[1, 0, 0], self.dst[2, 0, 0], self.dst[3, 0, 0]])
        rr, cc = dr.polygon(Y, X)
        color = [0,255,0]
        dr.set_color(img, [rr, cc], color, alpha=0.5)
        ax.imshow(img, plt.cm.gray)

        # draw center point
        ax.plot(self.computeCenter()[0], self.computeCenter()[1], 'k+', alpha=0.8)

        plt.title('Finded pattern in the scene [center:(%d, %d)]' % (self.computeCenter()[0], self.computeCenter()[1]))
        ax.axis('off')

        plt.subplots_adjust(left=0.05,  bottom=0.05,
                            right=0.95, top=0.9,
                            hspace=0.1, wspace=0.05)

