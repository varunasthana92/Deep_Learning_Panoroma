import cv2
import numpy as np
import scipy.ndimage.filters as sci
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage import data, img_as_float

img= cv2.imread('1.jpg')
# img=cv2.resize(img,((int)(img.shape[0]/3),(int)(img.shape[1]/3)))
imgCopy=np.copy(img)
imgCopy2=np.copy(img)
# # cv2.imshow('original',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows() 

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv2.cornerHarris(gray,4,3,0.04)

#result is dilated for marking the corners, not important
# dst = cv2.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.01*dst.max()]=[0,0,255]
# cv2.imwrite('CornerAll.jpg',img)
# cv2.imshow('Corner',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

h,w,c = img.shape

image_max = ndi.maximum_filter(gray, size=20, mode='constant')
lmCA = peak_local_max(gray, min_distance=20)

# localMax = sci.maximum_filter(dst, size=(h/5,w/5))
# uniqueLM= np.unique(localMax)
# lmCord = []
# # msk = (dst == localMax)*1
# for h_ in range(h):
# 	for w_ in range(w):
# 		if(dst[h_][w_] in uniqueLM):
# 			lmCord.append([h_,w_])
# lmCA=np.array(lmCord)
Nstrong= len(lmCA)
print('Nstrong= '+ str(Nstrong))
cv2.imwrite('CornerAll.jpg',img)
cv2.imshow('Corner',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
r=np.ones((Nstrong,1))*(float('inf'))

if(Nstrong>1000):
	Nbest=500
elif(Nstrong>600):
	Nbest=300

count =1
for i in range(Nstrong):
	if(i%500==0):
		print('Nstrong= '+str(Nstrong)+' and i= '+ str(i))
	for j in range(0,Nstrong):
		if(gray[lmCA[j][0]][lmCA[j][1]] > gray[lmCA[i][0]][lmCA[i][1]]):
			ed= (lmCA[j][0]-lmCA[i][0])**2 + (lmCA[j][1]-lmCA[i][1])**2 
		else:
			ed= float('inf')
		if(ed<r[i]):
			r[i]=ed
points=np.zeros((Nstrong,3), dtype='f')

for i in range(Nstrong):
	points[i][0]= r[i][0]
	points[i][1]= lmCA[i][0]
	points[i][2]= lmCA[i][1]

pointsSorted = points[points[:,0].argsort()[::-1]]
temp=pointsSorted[0:Nbest,1:3]

for i in range(Nbest):
	imgCopy[int(temp[i][0])][int(temp[i][1])]=[0,0,255]

# imgCopy[pointsSorted[0:500,1:3]]=[0,0,255]
cv2.imwrite('CornerANMS.jpg',imgCopy)
cv2.imshow('ANMS',imgCopy)
cv2.waitKey(0)
cv2.destroyAllWindows()



# r.sort(reverse=1)



# cv2.imshow('Harris',har)
# cv2.waitKey(0)
# cv2.destroyAllWindows() 
