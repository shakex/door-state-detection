import skimage
import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage import io
import cv2
from skimage.feature import *
from skimage.color import *
from skimage import data, exposure
from pylab import *
import functions as fun
from skimage.morphology import watershed, disk
from skimage.filters import rank
from skimage.util import img_as_ubyte
from scipy import ndimage as ndi


# -------------
# Import Frames
# -------------

'''
fileName = ['images/frame0.jpg', 'images/frame1.jpg',
			'images/frame2.jpg', 'images/frame3.jpg',
			'images/frame4.jpg', 'images/frame5.jpg',
			'images/frame6.jpg', 'images/frame7.jpg',
			'images/frame8.jpg', 'images/frame9.jpg',
			'images/frame10.jpg', 'images/frame11.jpg']

frame0_cv = cv2.imread(fileName[0])
frame1_cv = cv2.imread(fileName[1])
frame2_cv = cv2.imread(fileName[2])
frame3_cv = cv2.imread(fileName[3])
frame4_cv = cv2.imread(fileName[4])
frame5_cv = cv2.imread(fileName[5])
frame6_cv = cv2.imread(fileName[6])
frame7_cv = cv2.imread(fileName[7])
frame8_cv = cv2.imread(fileName[8])
frame9_cv = cv2.imread(fileName[9])
frame10_cv = cv2.imread(fileName[10])
frame11_cv = cv2.imread(fileName[11])

frame_cv = [frame0_cv, frame1_cv, frame2_cv,
			frame3_cv, frame4_cv, frame5_cv,
			frame6_cv, frame7_cv, frame8_cv,
			frame9_cv, frame10_cv, frame11_cv]

frame = io.imread_collection(fileName)

axrow, axcol = 2, 6
fig, ax = plt.subplots(axrow, axcol, figsize=(16, 8))
plt.gray()

for i in range(axrow):
	for j in range(axcol):
		ax[i, j].imshow(frame[i * axcol + j])
		ax[i, j].set_title("frame%d" % (i * axcol + j))

for ax in ax.ravel():
	ax.axis('off')

plt.subplots_adjust(left=0.05, bottom=0.05,
					right=0.95, top=0.95,
					hspace=0.1, wspace=0.05)

# plt.savefig('frames.png')
'''

# ---
# ORB
# ---

'''
kpt_orb = []
des_orb = []
matches_orb = []
descriptor_extractor = ORB(n_keypoints=80)
for i in range(len(frame)):
	descriptor_extractor.detect_and_extract(rgb2gray(frame[i]))
	kpt_orb.append(descriptor_extractor.keypoints)
	des_orb.append(descriptor_extractor.descriptors)

for i in range(len(frame)):
	matches_orb.append(match_descriptors(des_orb[0], des_orb[i],
					   cross_check=False))

axrow, axcol = 3, 4
fig, ax = plt.subplots(axrow, axcol, figsize=(30, 15))
plt.gray()

for i in range(axrow):
	for j in range(axcol):
		plot_matches(ax[i, j], frame[0], frame[i * axcol + j],
					 kpt_orb[0], kpt_orb[i * axcol + j],
					 matches_orb[i * axcol + j],
					 keypoints_color='k', matches_color='y',
					 only_matches=False)
		ax[i, j].set_title("ORB: frame0 vs.frame%d" % (i * axcol + j))

for ax in ax.ravel():
	ax.axis('off')

plt.subplots_adjust(left=0.05, bottom=0.05,
					right=0.95, top=0.95,
					hspace=0.1, wspace=0.05)

plt.savefig('figure_orb.png')


# score
# score_ORB = np.zeros(7)
# matches_ORB = [matches11_orb, matches12_orb, matches13_orb,
#                matches14_orb, matches15_orb, matches16_orb, matches17_orb]
# des_ORB = [des1_orb, des2_orb, des3_orb,
#            des4_orb, des5_orb, des6_orb, des7_orb]
# k = 0

# for m in matches_ORB:
#     for i in m:
#         for j in range(des1_orb.shape[1]):
#             if des1_orb[i[0]][j] != des_ORB[k][i[1]][j]:
#                 score_ORB[k] = score_ORB[k] + 1
#     score_ORB[k] = score_ORB[k] / m.shape[0]
#     k = k + 1

# print score_ORB


# normalization
# score_ORB = fun.normalize(score_ORB)

'''


# -------
# CENSURE
# -------

'''
# detector=CENSURE(min_scale=1,max_scale=7,mode='DoB',non_max_threshold=0.15,line_threshold=10)
# detector.detect(gframe1)
# kpt1_censure=detector.keypoints

# fig,ax=plt.subplots(1,2,figsize=(15,8))
# plt.gray()
# ax[0].imshow(frame1)
# ax[0].scatter(detector.keypoints[:, 1], detector.keypoints[:, 0],
#               2 ** detector.scales, facecolors='none', edgecolors='r')
# ax[0].set_title("Original Image")

# plt.show()

'''


# ----
# SIFT-match
# ----

kpt_sift = []
kpt_sift_np = []
des_sift = []
matches_sift = []
matches_sift_np = []
matches = []
good = []
dst = []
area = []
MIN_MATCH_COUNT = 4

pattern = cv2.imread("images/pattern.jpg")
scene1 = cv2.imread("images/scene1.jpg")
scene2 = cv2.imread("images/scene2.jpg")
scene3 = cv2.imread("images/scene3.jpg")
scene4 = cv2.imread("images/scene4.jpg")
scene5 = cv2.imread("images/scene5.jpg")


scene = [scene1, scene2, scene3, scene4, scene5]

sift = cv2.xfeatures2d.SIFT_create(1900)
matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), {})

kpt_pattern, des_pattern = sift.detectAndCompute(pattern, None)


for i in range(len(scene)):
    tmp1, tmp2 = sift.detectAndCompute(scene[i], None)
    kpt_sift.append(tmp1)
    des_sift.append(tmp2)
    matches_sift.append(matcher.knnMatch(des_pattern, des_sift[i], 2))
    matches.append(sorted(matches_sift[i], key=lambda x: x[0].distance))
    good.append([m1 for (m1, m2) in matches[i]
                 if m1.distance < 0.85 * m2.distance])

    kpt_sift_np.append(fun.kpt2np(kpt_sift[i]))
    kpt_pattern_np = fun.kpt2np(kpt_pattern)
    matches_sift_np.append(fun.match2np(good[i]))

    if len(good[i]) > MIN_MATCH_COUNT:
        src_pts = np.float32(
            [kpt_pattern[m.queryIdx].pt for m in good[i]]).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [kpt_sift[i][m.trainIdx].pt for m in good[i]]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        h, w = pattern.shape[:2]
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1],
                          [w - 1, 0]]).reshape(-1, 1, 2)
        dst.append(cv2.perspectiveTransform(pts, M))
    else:
        print(
            "Not enough matches are found - {}/{}".format(len(good[i]), MIN_MATCH_COUNT))


axrow, axcol = 3, 3
fig, ax = plt.subplots(axrow, axcol, figsize=(14, 8))
plt.gray()

for i in range(axrow):
    for j in range(axcol):
        if(i * axcol + j<len(scene)):
            plot_matches(ax[i, j], fun.bgr2rgb(pattern), fun.bgr2rgb(scene[i * axcol + j]),
                         kpt_pattern_np, kpt_sift_np[i * axcol + j],
                         matches_sift_np[i * axcol + j],
                         keypoints_color='k', matches_color='y',
                         only_matches=True)
            ax[i, j].axis('off')

center = np.zeros((len(scene), 2))
fig, ax = plt.subplots(1, 1)
for i in range(len(scene)):
	    ax.imshow(fun.bgr2rgb(scene[i]),
	              cmap=plt.cm.gray, alpha=0.4)
	    width = dst[i][2, 0, 0] - dst[i][0, 0, 0]
	    height = dst[i][2, 0, 1] - dst[i][0, 0, 1]
	    rect = plt.Rectangle((dst[i][0, 0, 0], dst[i][0, 0, 1]),
	                         width, height,
	                         edgecolor='k', facecolor='y', linewidth=1, alpha=0.5)
	    ax.add_patch(rect)

	    area.append(fun.computearea(dst[i]))
	    center[i][0], center[i][1] = fun.computercenter(dst[i])

	    ax.plot(center[i][0], center[i][1], 'r+', alpha=0.8)
	    ax.axis('off')

# plt.subplots_adjust(left=0.05, bottom=0.05,
#                     right=0.95, top=0.95,
#                     hspace=0.1, wspace=0.05)

fig, ax = plt.subplots()
ax.plot([1, 2, 3, 4, 5], area, '.-')
plt.xlabel('the i-th scene')
plt.ylabel('Areas')
plt.title('Changes of areas')

print(dst[0])

'''
MIN_MATCH_COUNT = 4

pattern = cv2.imread("images/pattern.jpg")
scene1 = cv2.imread("images/scene5.jpg")

sift = cv2.xfeatures2d.SIFT_create(1900)

matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5),{})

kpt1, des1 = sift.detectAndCompute(pattern, None)
kpt2, des2 = sift.detectAndCompute(scene1, None)

matches = matcher.knnMatch(des1, des2, 2)

matches = sorted(matches, key=lambda x:x[0].distance)

good = [m1 for (m1, m2) in matches if m1.distance < 0.85* m2.distance]

canvas = scene1.copy()

if len(good) > MIN_MATCH_COUNT:
	src_pts = np.float32([ kpt1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
	dst_pts = np.float32([ kpt2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

	M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

	h,w = pattern.shape[:2]
	pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
	dst = cv2.perspectiveTransform(pts,M)
	cv2.polylines(canvas,[np.int32(dst)],True,(0,255,0),3, cv2.LINE_AA)
else:
	print( "Not enough matches are found - {}/{}".format(len(good),MIN_MATCH_COUNT))

# (8) drawMatches
matched = cv2.drawMatches(pattern,kpt1,canvas,kpt2,good,None)#,**draw_params)

# (9) Crop the matched region from scene
h,w = pattern.shape[:2]
pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
dst = cv2.perspectiveTransform(pts,M)
perspectiveM = cv2.getPerspectiveTransform(np.float32(dst),pts)
found = cv2.warpPerspective(scene1, perspectiveM,(w,h))

cv2.imshow("matched", matched)
cv2.imshow("found", found)
cv2.imwrite('scene5.png',matched)

cv2.waitKey();cv2.destroyAllWindows()
'''

'''
kpt_sift = []
kpt_sift_np = []
des_sift = []
matches_sift = []
matches_sift_np = []

sift = cv2.xfeatures2d.SIFT_create(50)
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

for i in range(len(frame_cv)):
	tmp1, tmp2 = sift.detectAndCompute(frame_cv[i], None)
	kpt_sift.append(tmp1)
	des_sift.append(tmp2)
	matches_sift.append(bf.match(des_sift[0], des_sift[i]))
	kpt_sift_np.append(fun.kpt2np(kpt_sift[i]))
	matches_sift_np.append(fun.match2np(matches_sift[i]))

axrow, axcol = 3, 4
fig, ax = plt.subplots(axrow, axcol, figsize=(16, 8))
plt.gray()

for i in range(axrow):
	for j in range(axcol):
		plot_matches(ax[i, j], frame[0], frame[i * axcol + j],
					 kpt_sift_np[0], kpt_sift_np[i * axcol + j],
					 matches_sift_np[i * axcol + j],
					 keypoints_color='k', matches_color='y',
					 only_matches=False)
		ax[i, j].set_title("SIFT: frame0 vs.frame%d" % (i * axcol + j))

for ax in ax.ravel():
	ax.axis('off')

plt.subplots_adjust(left=0.05, bottom=0.05,
					right=0.95, top=0.95,
					hspace=0.1, wspace=0.05)

# plt.savefig('figure_sift.png')
'''

# ----
# surf
# ----

'''
kpt_surf = []
kpt_surf_np = []
des_surf = []
matches_surf = []
matches_surf_np = []

surf = cv2.xfeatures2d.SURF_create(7000)
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

for i in range(len(frame_cv)):
	tmp1, tmp2 = surf.detectAndCompute(frame_cv[i], None)
	kpt_surf.append(tmp1)
	des_surf.append(tmp2)
	matches_surf.append(bf.match(des_surf[0], des_surf[i]))
	kpt_surf_np.append(fun.kpt2np(kpt_surf[i]))
	matches_surf_np.append(fun.match2np(matches_surf[i]))

axrow, axcol = 3, 4
fig, ax = plt.subplots(axrow, axcol, figsize=(30, 15))
plt.gray()

for i in range(axrow):
	for j in range(axcol):
		plot_matches(ax[i, j], frame[0], frame[i * axcol + j],
					 kpt_surf_np[0], kpt_surf_np[i * axcol + j],
					 matches_surf_np[i * axcol + j],
					 keypoints_color='k', matches_color='y',
					 only_matches=False)
		ax[i, j].set_title("SURF: frame0 vs.frame%d" % (i * axcol + j))

for ax in ax.ravel():
	ax.axis('off')

plt.subplots_adjust(left=0.05, bottom=0.05,
					right=0.95, top=0.95,
					hspace=0.1, wspace=0.05)

plt.savefig('figure_surf.png')


# print gframe1.shape
# print kpt1_np
# print kpt2_np
# print matches12_np
# print kpt1_orb
# print matches12_orb


# print kpt1_np.shape, kpt1_np.dtype
# print matches12_np.shape, matches12_np.dtype
# print kpt1_orb.shape, kpt1_orb.dtype
# print matches12_orb.shape, matches12_orb.dtype

# print len(kpt1_sift), len(matches12_sift)
# print kpt1_sift[0].pt[0],kpt1_sift[0].pt[1]
# print matches12_sift
# print matches12_sift[0][0].queryIdx, matches12_sift[0][0].trainIdx

'''


# --------
# HOG
# --------

'''
ori = 4
ppc = (16, 16)
cpb = (4, 4)
des_hog = []
hog_frame = []
hog_frame_r = []
score_hog = []

for i in range(len(frame)):
	tmp1, tmp2 = hog(rgb2gray(frame[i]),
					 orientations=ori,
					 pixels_per_cell=ppc,
					 cells_per_block=cpb,
					 visualise=True)
	des_hog.append(tmp1)
	hog_frame.append(tmp2)
	hog_frame_r.append(exposure.rescale_intensity(hog_frame[i], in_range=(0, 0.02)))
	score_hog.append(sum((des_hog[i] - des_hog[0]) ** 2))

# score_hog=np.array([score_1,score_2,score_3,score_4,score_5,score_6,score_7])
# score_hog=normalize(score_hog)

axrow, axcol = 3, 4
fig, ax = plt.subplots(axrow, axcol, figsize=(30, 15))
plt.gray()

for i in range(axrow):
	for j in range(axcol):
		ax[i, j].imshow(hog_frame_r[i * axcol + j], cmap=plt.cm.gray)
		ax[i, j].set_title("HoG: frame%d (score:%.2f)" % (i * axcol + j, score_hog[i * axcol + j]))

for ax in ax.ravel():
	ax.axis('off')

plt.subplots_adjust(left=0.05, bottom=0.05,
					right=0.95, top=0.95,
					hspace=0.1, wspace=0.05)

plt.savefig('figure_hog.png')

'''


# ------------
# WatershedSeg
# ------------

'''
denoised = []
gradient_frame = []
markers = []
labels = []
frame_ubyte = []

for i in frame:
	frame_ubyte.append(img_as_ubyte(i))

for i in range(len(frame)):
	denoised.append(rank.median(rgb2gray(frame_ubyte[i]), disk(2)))
	tmp1 = rank.gradient(denoised[i], disk(5)) < 10
	tmp1 = ndi.label(tmp1)[0]
	markers.append(tmp1)
	gradient_frame.append(rank.gradient(denoised[i], disk(2)))
	labels.append(watershed(gradient_frame[i], markers[i],
				  watershed_line=True))

axrow, axcol = 2, 6
fig, ax = plt.subplots(axrow, axcol, figsize=(20, 10))
plt.gray()

for i in range(axrow):
	for j in range(axcol):
		ax[i, j].imshow(gradient_frame[i * axcol + j], cmap=plt.cm.spectral, interpolation='nearest')
		ax[i, j].set_title("Local gradient: frame%d" % (i * axcol + j))

for ax in ax.ravel():
	ax.axis('off')

plt.subplots_adjust(left=0.05, bottom=0.05,
					right=0.95, top=0.95,
					hspace=0.1, wspace=0.05)

plt.savefig('figure_gradient.png')


axrow, axcol = 2, 6
fig, ax = plt.subplots(axrow, axcol, figsize=(20, 10))
plt.gray()

for i in range(axrow):
	for j in range(axcol):
		ax[i, j].imshow(rgb2gray(frame[i * axcol + j]), cmap=plt.cm.gray, interpolation='nearest')
		ax[i, j].imshow(labels[i * axcol + j], cmap=plt.cm.spectral, interpolation='nearest', alpha=.7)
		ax[i, j].set_title("segmented: frame%d" % (i * axcol + j))

for ax in ax.ravel():
	ax.axis('off')

plt.subplots_adjust(left=0.05, bottom=0.05,
					right=0.95, top=0.95,
					hspace=0.1, wspace=0.05)

plt.savefig('figure_labels.png')

'''


plt.show()
