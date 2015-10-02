from numpy import *
from tensor_personal_functions import *

#http://sebastianraschka.com/Articles/2014_matlab_vs_numpy.html

#Generate a tensor, any order, any size
#Value is the default value, commonly 0
def initTensor(value, *lengths):
	list = []
	dim = len(lengths)
	if dim == 1:
		for i in range(lengths[0]):
			list.append(value)
	elif dim > 1:
		for i in range(lengths[0]):
			list.append(initTensor(value, *lengths[1:]))
	return list

#Tenseur alternateur
def generate_epsilon_ijk():
	epsilon_ijk = initTensor( 0., 3, 3, 3)
	
	for i in range(0, len(epsilon_ijk) ):
		for j in range(0, len(epsilon_ijk[0]) ):
			for k in range(0, len(epsilon_ijk[0][0]) ):
				
				if (i == j) or (j == k) or (i == k):
					epsilon_ijk[i][j][k] = 0.
				if (`i`+`j`+`k` == "012") or (`i`+`j`+`k` == "120") or (`i`+`j`+`k` == "201"):
					epsilon_ijk[i][j][k] = 1.
				if (`i`+`j`+`k` == "021") or (`i`+`j`+`k` == "102") or (`i`+`j`+`k` == "210"):
					epsilon_ijk[i][j][k] = -1.
			
	return epsilon_ijk

	
def kronecker( i, j ):
	if ( i == j ):
		return 1.
	else:
		return 0.
		
def indentity12_12():
	I = initTensor(0., 12, 12)
	
	for i in range(0, 12):
		I[i][i] = 1.
		
	return I


		

	
