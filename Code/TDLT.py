import glob
import cv2
import numpy as np 
import random

def TDLT(ca, h4pt):
	A = []
	B = []

	for i in range(4):
		x, y = ca[i][0], ca[i][1]
		u, v = h4pt[i], h4pt[i+5]
		B.append(-v)
		B.append(u)
		tempA1 = [0 0 0 -x -y -1 x*v y*v]
		tempA2 = [x y 1 0 0 0 -x*u -y*u]
		A.append(tempA1)
		A.append(tempA2)

	Ainv = np.linalg.pinv(np.array(A))
	H = np.matmul(Ainv, np.array(B))
	return H