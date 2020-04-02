import sys
from scipy.spatial import distance
import numpy as np
#http://www.scipy.org/
try:
	from numpy import dot
	from numpy.linalg import norm
except:
	print("Error: Requires numpy from http://www.scipy.org/. Have you installed scipy?")
	sys.exit() 

def removeDuplicates(list):
	""" remove duplicates from a list """
	return set((item for item in list))


def cosine(vector1, vector2):
	""" related documents j and q are in the concept space by comparing the vectors :
		cosine  = ( V1 * V2 ) / ||V1|| x ||V2|| """
	return float(dot(vector1,vector2) / (norm(vector1) * norm(vector2)))

def euclidean(vector1, vector2):
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)
    # distance.euclidean(vector1, vector2)
    return norm(vector1 - vector2)

def euc_distance(alist, blist):
    sum_of = 0
    for x, y in zip(alist, blist):
        ans = (x - y)**2
        sum_of += ans
    return float(sum_of)**(1/2)
