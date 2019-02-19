import os
import os.path
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import skimage.draw as dr
from featureMatch import SIFTMatch
from featureMatch import SURFMatch
from featureMatch import ORBMatch
import featureMatch._functions as fun
import skdemo
from skimage.color import rgb2gray
from scipy.optimize import curve_fit


pattern = io.ImageCollection('images/pattern/*.JPG')
scene1_1 = io.ImageCollection('images/scene1_1/*.JPG')
scene1_1_na = io.ImageCollection('images/scene1_1_noAngle/*.png')
scene1_2 = io.ImageCollection('images/scene1_2/*.JPG')
scene1_2_na = io.ImageCollection('images/scene1_2_noAngle/*.png')
scene2_1 = io.ImageCollection('images/scene2_1/*.JPG')
scene2_1_na = io.ImageCollection('images/scene2_1_noAngle/*.png')
scene2_2 = io.ImageCollection('images/scene2_2/*.JPG')
scene2_2_na = io.ImageCollection('images/scene2_2_noAngle/*.png')
scene3_3 = io.ImageCollection('images/scene3_3/*.JPG')
scene4_4 = io.ImageCollection('images/scene4_4/*.JPG')
scene4_4_na = io.ImageCollection('images/scene4_4_noAngle/*.png')
scene4_5 = io.ImageCollection('images/scene4_5/*.JPG')

angle4_4 = np.arange(0, 95, 5)
angle4_5 = np.arange(0, 95, 5)
angle2_2 = np.arange(0, 75, 10)
angle2_1 = np.arange(0, 75, 10)
angle1_2 = np.arange(0, 95, 10)
angle1_1 = np.arange(0, 95, 10)


# SIFT
# ----

# pattern = cv2.imread("images/pattern/pattern4.jpg")
# scene = cv2.imread("images/scene/scene4_4-088.png")

# sift = SIFTMatch()
# sift.init(pattern, scene, num_kpts = 1000, ratio = 0.84, min_match_count = 4)
# sift.siftCompute()

# area = sift.computeArea()
# center = sift.computeCenter()
# sift.plotMatch()
# sift.plotFind()

# Settings
# --------
sceneAnalyse = scene4_4_na
patternAnalyse = pattern[3]
angle = angle4_4


kptNumList = []
matchNumList = []
areaList = []
centerList = []
dstList = []

# compute the first scene
# -----------------------
sift = SIFTMatch()
sift.init(patternAnalyse, sceneAnalyse[0], num_kpts = 800, ratio = 0.84, min_match_count = 4)
sift.siftCompute()
area_0 = sift.computeArea()
center_0 = sift.computeCenter()

for idx, i in enumerate(sceneAnalyse):

    # sift compute
    sift = SIFTMatch()
    sift.init(patternAnalyse, i, num_kpts = 800, ratio = 0.84, min_match_count = 4)
    sift.siftCompute()

    # compute area&center
    area = sift.computeArea()
    center = sift.computeCenter()

    # remove bad detection
    if (area > sceneAnalyse[0].shape[0] * sceneAnalyse[0].shape[1]) or (center[0] < 0 or center[0] > sceneAnalyse[0].shape[1]) or (center[1] < 0 or center[1] > sceneAnalyse[0].shape[0]):
        area = 0
        center = [0, 0]

    # save(#keypoints,#matches,dist,area,center) to lists
    kptNumList.append(sift.num_kpts)
    matchNumList.append(sift.matches.shape[0])
    areaList.append(area)
    centerList.append(center)
    dstList.append(sift.dst)

last_matchNum = 1
for i in np.arange(1, len(areaList), 1):
	if(abs(areaList[i] - areaList[i-1]) > area_0 * 0.1):
		last_matchNum = i
		break

print(last_matchNum)


# Plot detected
# -------------

colNum = 8
rowNum = int(len(sceneAnalyse) / colNum) + 1
# 1.6 # 4
fig, ax = plt.subplots(rowNum, colNum, figsize=(16, int(1.6 * rowNum)))
for idx, i in enumerate(sceneAnalyse):
    ax[int(idx / colNum), idx % colNum].imshow(i, plt.cm.gray)

    width = dstList[idx][2, 0, 0] - dstList[idx][0, 0, 0]
    height = dstList[idx][2, 0, 1] - dstList[idx][0, 0, 1]

    if(dstList[idx][0, 0, 0] > 0 and dstList[idx][0, 0, 0] < sceneAnalyse[0].shape[1] and dstList[idx][0, 0, 1] > 0 and dstList[idx][0, 0, 1] < sceneAnalyse[0].shape[0]):
        rect = plt.Rectangle((dstList[idx][0, 0, 0], dstList[idx][0, 0, 1]), width, height,
                            edgecolor='k', facecolor='y', linewidth=1, alpha=0.6)
        ax[int(idx / colNum), idx % colNum].add_patch(rect)
        ax[int(idx / colNum), idx % colNum].plot(centerList[idx][0], centerList[idx][1], 'k+', alpha=1)

    if(len(sceneAnalyse) < 20):
        ax[int(idx / colNum), idx % colNum].set_title("%d degree" % angle[idx])
    else:
        ax[int(idx / colNum), idx % colNum].set_title("#%d" % idx)

for ax in ax.ravel():
	ax.axis('off')

plt.subplots_adjust(left=0.05, bottom=0.05,
					right=0.95, top=0.95,
					hspace=0, wspace=0)

# plt.savefig('find_4-5a.png')

# curve fit
# ---------
x = np.zeros(len(centerList))
y = np.zeros(len(centerList))
for idx, i in enumerate(centerList):
    x[idx] = centerList[idx][0]
    y[idx] = centerList[idx][1]

# for angle-labeled sample
if(len(sceneAnalyse) < 20):
    z_x = np.polyfit(angle[0:last_matchNum], x[0:last_matchNum], 2)
    z_y = np.polyfit(angle[0:last_matchNum], y[0:last_matchNum], 2)
    p_x = np.poly1d(z_x)
    p_y = np.poly1d(z_y)
    x_pred = p_x(angle[0:last_matchNum])
    y_pred = p_y(angle[0:last_matchNum])

# for no-angle-labeled sample
else:
    z_x = np.polyfit(range(len(areaList[0:last_matchNum])), x[0:last_matchNum], 3)
    z_y = np.polyfit(range(len(areaList[0:last_matchNum])), y[0:last_matchNum], 3)
    p_x = np.poly1d(z_x)
    p_y = np.poly1d(z_y)
    x_pred = p_x(range(len(areaList[0:last_matchNum])))
    y_pred = p_y(range(len(areaList[0:last_matchNum])))

plt.figure(figsize=(9, 6))

plt.subplot(221)
if(len(sceneAnalyse) < 20):
    plt.scatter(angle[0:last_matchNum], x[0:last_matchNum], facecolor="none", edgecolor="b", s=30, label="(angle, x)")
    plt.plot(angle[0:last_matchNum], x_pred, 'r', label="prediction")
    plt.title('Angle vs Center(X-axis)')
else:
    plt.scatter(range(len(areaList[0:last_matchNum])), x[0:last_matchNum], facecolor="none", edgecolor="b", s=30, label="(#frame, x)")
    plt.plot(range(len(areaList[0:last_matchNum])), x_pred, 'r', label="prediction")
    plt.title('Frame# vs Center(X-axis)')
plt.legend()

plt.subplot(222)
if(len(sceneAnalyse) < 20):
    plt.scatter(angle[0:last_matchNum], y[0:last_matchNum], facecolor="none", edgecolor="b", s=30, label="(angle, y)")
    plt.plot(angle[0:last_matchNum], y_pred, 'r', label="prediction")
    plt.title('Angle vs Center(Y-axis)')
else:
    plt.scatter(range(len(areaList[0:last_matchNum])), y[0:last_matchNum], facecolor="none", edgecolor="b", s=30, label="(#frame, y)")
    plt.plot(range(len(areaList[0:last_matchNum])), y_pred, 'r', label="prediction")
    plt.title('Frame# vs Center(Y-axis)')
plt.legend()


# for no-angle-labeled samples: plot(frame# - Area)
# -----------------------------------------------
if(len(sceneAnalyse) < 20):
    z_a = np.polyfit(angle[0:last_matchNum], areaList[0:last_matchNum], 3)
    p_a = np.poly1d(z_a)
    a_pred = p_a(angle[0:last_matchNum])
else:
    z_a = np.polyfit(range(len(areaList[0:last_matchNum])), areaList[0:last_matchNum], 3)
    p_a = np.poly1d(z_a)
    a_pred = p_a(range(len(areaList[0:last_matchNum])))


plt.subplot(223)
if(len(sceneAnalyse) < 20):
    plt.scatter(angle, areaList, facecolor="none", edgecolor="b", s=10)
    plt.title('Angle vs Area of objects')
else:
    plt.scatter(range(len(areaList)), areaList, facecolor="none", edgecolor="b", s=10)
    plt.title('Frame# vs Area of objects')
plt.grid()

plt.subplot(224)
if(len(sceneAnalyse) < 20):
    plt.scatter(angle[0:last_matchNum], areaList[0:last_matchNum], facecolor="none", edgecolor="b", s=10, label="training data")
    plt.plot(angle[0:last_matchNum], a_pred, 'r', label="prediction")
    plt.title('Angle vs Area of objects (Find %d/%d)' % (last_matchNum, len(areaList)))
else:
    plt.scatter(range(len(areaList[0:last_matchNum])), areaList[0:last_matchNum], facecolor="none", edgecolor="b", s=10, label="training data")
    plt.plot(range(len(areaList[0:last_matchNum])), a_pred, 'r', label="prediction")
    plt.title('Frame# vs Area of objects (Find %d/%d)' % (last_matchNum, len(areaList)))
plt.legend()

plt.subplots_adjust(hspace=0.35, wspace=0.35)
# plt.savefig('analyse_4-5a.png')

# for angle-labeled samples: plot(angle - Area)
# ---------------------------------------------
# angleRange = np.arange(0,95,5)
# fig, ax = plt.subplots()
# ax.plot(angleRange[0:15], areaList[0:15], '.-')
# plt.xlabel('door open angle')
# plt.ylabel('the area of detected pattern')
# plt.title('Relationship between Angle - Area')


plt.show()

