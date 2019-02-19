import cv2
import numpy as np
import matplotlib.pyplot as plt
from featureMatch import SIFTMatch
from featureMatch import SURFMatch
from featureMatch import ORBMatch
import featureMatch._functions as fun

pattern = cv2.imread("images/pattern/pattern2.jpg")
scene = cv2.imread("images/scene2_2_noAngle/scene2_2-001.png")


# orb = ORBMatch()
# orb.init(pattern, scene, num_kpts = 800, min_match_count = 4)
# orb.orbCompute()
# area = orb.computeArea()
# center = orb.computeCenter()

# print(orb.kpts1.shape[0])
# print(orb.kpts2.shape[0])
# print(orb.matches.shape[0])
# print(orb.dst)

# print(area)
# print(center)

# orb.plotMatch()
# orb.plotFind()

sift = SIFTMatch()
sift.init(pattern, scene, num_kpts = 1500, ratio = 0.84, min_match_count = 4)
sift.siftCompute()

area = sift.computeArea()
center = sift.computeCenter()
sift.plotMatch()
sift.plotFind()
print(sift.kpts1.shape[0])
print(sift.kpts2.shape[0])

print(area)
print(center)

# dst = np.zeros((4, 1, 2))
# dst[0,0,0] = 580
# dst[0,0,1] = 146
# dst[1,0,0] = 599
# dst[1,0,1] = 165
# dst[2,0,0] = 597
# dst[2,0,1] = 273
# dst[3,0,0] = 578
# dst[3,0,1] = 146
# x, y = fun.computercenter(dst)
# area = fun.computearea(dst)
# print(x,y)
# print(area)


# surf = SURFMatch()
# surf.init(pattern, scene, hessianThreshold = 800, ratio = 0.83)
# surf.surfCompute()

# print("# of kpts1: %d" % surf.kpts1.shape[0])
# print("# of kpts2: %d" % surf.kpts2.shape[0])
# print("# of matches: %d" % surf.matches.shape[0])
# print(surf.dst)
# color=[0, 255, 0]
# surf.plotFind()

# plt.show()

# Args choose test (SURF)
# -----------------------
# h_class = range(800,7000,200)
# r_class = np.arange(0.65, 0.95, 0.01)
# h_len = len(h_class)
# r_len = len(r_class)

# surf = []
# gt_area = 20451
# gt_center = np.array([446, 244])
# area_good = []
# center_good = []
# para_h = []
# para_r = []

# for i, h in enumerate(h_class):
# 	for j, r in enumerate(r_class):
# 		surf.append(SURFMatch())
# 		surf[i * r_len + j].init(pattern, scene, hessianThreshold = h, ratio = r)
# 		surf[i * r_len + j].surfCompute()

# 		area = surf[i * r_len + j].computeArea()
# 		center = surf[i * r_len + j].computeCenter()

# 		if area<=18388 and area>=15388 and center[0]<=508 and center[0]>=468 and center[1]<=237 and center[1]>=197:
# 			area_good.append(area)
# 			center_good.append(center)
# 			para_h.append(h)
# 			para_r.append(r)

# 		# print("#%d (h=%d, r=%f)" % ((i * r_len + j + 1), surf[i * r_len + j].hessianThreshold, surf[i * r_len + j].ratio))
# 		# print("Number of matched: %d" % surf[i * r_len + j].matches.shape[0])
# 		# print("Area: %.2f" % area)
# 		# print("Center: (%.2f, %.2f)" % (center[0], center[1]))

# for i in range(len(area_good)):
# 	print("#%d: (h=%d, r=%.2f)" % (i+1, para_h[i], para_r[i]))
# 	print("Area = %.2f" % area_good[i])
# 	print("Center = (%.2f, %.2f)" % (center_good[i][0], center_good[i][1]))

# plt.show()



# Args choose test (SIFT)
# -----------------------
# n_class = range(50, 800, 50)
# r_class = np.arange(0.65, 0.95, 0.01)
# n_len = len(n_class)
# r_len = len(r_class)

# sift = []
# gt_area = 4238
# gt_center = np.array([580, 148])
# area_good = []
# center_good = []
# para_n = []
# para_r = []

# for i, n in enumerate(n_class):
# 	for j, r in enumerate(r_class):
# 		sift.append(SIFTMatch())
# 		sift[i * r_len + j].init(pattern, scene, num_kpts = n, ratio = r)
# 		sift[i * r_len + j].siftCompute()

# 		area = sift[i * r_len + j].computeArea()
# 		center = sift[i * r_len + j].computeCenter()

# 		if area<=gt_area+500 and area>=gt_area-500 and center[0]<=gt_center[0]+10 and center[0]>=gt_center[0]-10 and center[1]<=gt_center[1]+10 and center[1]>=gt_center[1]-10:
# 			area_good.append(area)
# 			center_good.append(center)
# 			para_n.append(n)
# 			para_r.append(r)

# 		# print("#%d (h=%d, r=%f)" % ((i * r_len + j + 1), surf[i * r_len + j].hessianThreshold, surf[i * r_len + j].ratio))
# 		# print("Number of matched: %d" % surf[i * r_len + j].matches.shape[0])
# 		# print("Area: %.2f" % area)
# 		# print("Center: (%.2f, %.2f)" % (center[0], center[1]))

# for i in range(len(area_good)):
# 	print("#%d: (h=%d, r=%.2f)" % (i+1, para_n[i], para_r[i]))
# 	print("Area = %.2f" % area_good[i])
# 	print("Center = (%.2f, %.2f)" % (center_good[i][0], center_good[i][1]))


plt.show()


