from numpy import *
from projectors_personal_functions import *

#=============================================================
# Symmetries
#=============================================================
#	check_tensor_minor_symmetry( tensor )
#	check_tensor_major_symmetry( tensor )
#	apply_minor_sym_to_tensor_term( A_tensor4, i, j, k, l )
#	apply_major_sym_to_tensor_term( A_tensor4, i, j, k, l )

#=============================================================
# Voigt
#=============================================================
#	tensor4_to_voigt4( A_tensor4 )
#	voigt4_to_tensor4( A_voigt4 )

#=============================================================
# Base change
#=============================================================
#	tensorial_base_change( P, tensorA )

#=============================================================
# Matrix
#=============================================================
#	matrix_dot_matrix( matrixa, matrixb )

def check_tensor_minor_symmetry( tensor ):
	for i in range(len(tensor[0][0][0])):
		for j in range(len(tensor[0][0][0])):
			for k in range(len(tensor[0][0][0])):
				for l in range(len(tensor[0][0][0])):
					
					if (tensor[i][j][k][l] != tensor[j][i][k][l]):
						print "[check_tensor_minor_symmetry] Tensor is not symmetrical "
						print i,j,k,l, tensor[i][j][k][l], tensor[j][i][k][l]
						return False
						
					if (tensor[i][j][k][l] != tensor[i][j][l][k]):
						print "[check_tensor_minor_symmetry] Tensor is not symmetrical "
						print i,j,k,l, tensor[i][j][k][l], tensor[i][j][l][k]
						return False
						
					if (tensor[i][j][k][l] != tensor[j][i][l][k]):
						print "[check_tensor_minor_symmetry] Tensor is not symmetrical "
						print i,j,k,l, tensor[i][j][k][l], tensor[j][i][l][k]
						return False
	return True


def check_tensor_major_symmetry( tensor ):
	for i in range(len(tensor[0][0][0])):
		for j in range(len(tensor[0][0][0])):
			for k in range(len(tensor[0][0][0])):
				for l in range(len(tensor[0][0][0])):
					
					if (tensor[i][j][k][l] != tensor[k][l][i][j]):
						print "[check_tensor_major_symmetry] Tensor is not symmetrical "
						print i,j,k,l, tensor[i][j][k][l], tensor[k][l][i][j]
						return False
	return True


def tensor4_to_voigt4( A_tensor4 ):
	A_voigt4 = initTensor(0., 6, 6)
	
	#blue
	for i in range( 0, 3 ):
		for j in range( 0, 3 ):
			A_voigt4[i][j] = A_tensor4[i][i][j][j]
		

	for i in range( 0, 3 ):
		#print i
		A_voigt4[3][i] = sqrt(2) * A_tensor4[1][2][i][i]
		A_voigt4[4][i] = sqrt(2) * A_tensor4[2][0][i][i]
		A_voigt4[5][i] = sqrt(2) * A_tensor4[0][1][i][i]
		
	for j in range( 0, 3 ):
		#print j
		A_voigt4[j][3] = sqrt(2) * A_tensor4[j][j][1][2]
		A_voigt4[j][4] = sqrt(2) * A_tensor4[j][j][2][0]
		A_voigt4[j][5] = sqrt(2) * A_tensor4[j][j][0][1]
		
	A_voigt4[3][3] = 2 * A_tensor4[1][2][1][2]
	A_voigt4[4][3] = 2 * A_tensor4[2][0][1][2]
	A_voigt4[5][3] = 2 * A_tensor4[0][1][1][2]
	
	A_voigt4[3][4] = 2 * A_tensor4[1][2][2][0]
	A_voigt4[4][4] = 2 * A_tensor4[2][0][2][0]
	A_voigt4[5][4] = 2 * A_tensor4[0][1][2][0]

	A_voigt4[3][5] = 2 * A_tensor4[1][2][0][1]
	A_voigt4[4][5] = 2 * A_tensor4[2][0][0][1]
	A_voigt4[5][5] = 2 * A_tensor4[0][1][0][1]
	
	#print "AVOIGT4[0]:"
	#for i in range(len(A_voigt4[0])):
		#print A_voigt4[i]

	return A_voigt4


#Takes A_voigt(6,6) gives back A_tensor(4,4,4,4), with symmetries
def voigt4_to_tensor4( A_voigt4 ):

	A_tensor4 = initTensor(0., 3, 3, 3, 3)
	
	a = 0
	b = 0
	A_voigt4_length = len( A_voigt4 )
	for a in range(0, A_voigt4_length ):
		for b in range(0, A_voigt4_length ):

			flaga = False
			flagb = False
			
			if (a == 0):
				i = 0
				j = 0
				
			if (b == 0):
				k=0
				l=0
				
			if (a == 1):
				i=1
				j=1
				
			if (b == 1):
				k=1
				l=1
				
			if (a == 2):
				i=2
				j=2

			if (b == 2):
				k=2
				l=2
				
				
			if (a == 3):
				i=0
				j=1
				ip=1
				jp=2
				flaga= True

			if (b ==3):
				k=0
				l=1
				kp=1
				lp=2
				flagb= True	

			if (a == 4):
				i=1
				j=2
				ip=2
				jp=0
				flaga= True
	
			if (b == 4):
				k=1
				l=2
				kp=2
				lp=0
				flagb= True
	
			if (a == 5):
				i=0
				j=2
				ip=0
				jp=1
				flaga= True

			if (b == 5):
				k=0
				l=2
				kp=0
				lp=1
				flagb = True
			
			
			if (flaga is True) and (flagb is True):
				A_tensor4[ip][jp][kp][lp] = A_voigt4[a][b]/2.
				A_tensor4 = apply_minor_sym_to_tensor_term( A_tensor4, ip, jp, kp, lp )
				A_tensor4 = apply_major_sym_to_tensor_term( A_tensor4, ip, jp, kp, lp )
			if (flaga is True) and (flagb is False):
				A_tensor4[ip][jp][k][l] = A_voigt4[a][b]/(sqrt(2.))
				A_tensor4 = apply_minor_sym_to_tensor_term( A_tensor4, ip, jp, k, l )
				A_tensor4 = apply_major_sym_to_tensor_term( A_tensor4, ip, jp, k, l )
			if(flagb is True) and (flaga is False):
				A_tensor4[i][j][kp][lp] = A_voigt4[a][b]/(sqrt(2.))
				A_tensor4 = apply_minor_sym_to_tensor_term( A_tensor4, i, j, kp, lp )
				A_tensor4 = apply_major_sym_to_tensor_term( A_tensor4, i, j, kp, lp )
				
			if (flaga is False) and (flagb is False):
			
				A_tensor4[i][j][k][l] = A_voigt4[a][b]
				A_tensor4 = apply_minor_sym_to_tensor_term( A_tensor4, i, j, k, l )
				A_tensor4 = apply_major_sym_to_tensor_term( A_tensor4, i, j, k, l )
			
			#print "A_voigt4[",a ,"][", b, "] = " , "A_tensor4[", i, "][", j, "][", k, "][", l, "]" 
			#print A_tensor4[i][j][k][l], A_voigt4[a][b]
			#print "-------------------------------------------------------------------"

	return A_tensor4
	
#Takes A_voigt(6,6) gives back A_tensor(4,4,4,4), with symmetries
def voigt4_to_tensor4_no_symmetry( A_voigt4 ):

	A_tensor4 = initTensor(0., 3, 3, 3, 3)
	
	a = 0
	b = 0
	A_voigt4_length = len( A_voigt4 )
	for a in range(0, A_voigt4_length ):
		for b in range(0, A_voigt4_length ):

			flaga = False
			flagb = False
			
			if (a == 0):
				i = 0
				j = 0
				
			if (b == 0):
				k=0
				l=0
				
			if (a == 1):
				i=1
				j=1
				
			if (b == 1):
				k=1
				l=1
				
			if (a == 2):
				i=2
				j=2

			if (b == 2):
				k=2
				l=2
				
				
			if (a == 3):
				i=0
				j=1
				ip=1
				jp=2
				flaga= True

			if (b ==3):
				k=0
				l=1
				kp=1
				lp=2
				flagb= True	

			if (a == 4):
				i=1
				j=2
				ip=2
				jp=0
				flaga= True
	
			if (b == 4):
				k=1
				l=2
				kp=2
				lp=0
				flagb= True
	
			if (a == 5):
				i=0
				j=2
				ip=0
				jp=1
				flaga= True

			if (b == 5):
				k=0
				l=2
				kp=0
				lp=1
				flagb = True
			
			
			if (flaga is True) and (flagb is True):
				A_tensor4[ip][jp][kp][lp] = A_voigt4[a][b]/2.
				#A_tensor4 = apply_minor_sym_to_tensor_term( A_tensor4, ip, jp, kp, lp )
				#A_tensor4 = apply_major_sym_to_tensor_term( A_tensor4, ip, jp, kp, lp )
			if (flaga is True) and (flagb is False):
				A_tensor4[ip][jp][k][l] = A_voigt4[a][b]/(sqrt(2.))
				#A_tensor4 = apply_minor_sym_to_tensor_term( A_tensor4, ip, jp, k, l )
				#A_tensor4 = apply_major_sym_to_tensor_term( A_tensor4, ip, jp, k, l )
			if(flagb is True) and (flaga is False):
				A_tensor4[i][j][kp][lp] = A_voigt4[a][b]/(sqrt(2.))
				#A_tensor4 = apply_minor_sym_to_tensor_term( A_tensor4, i, j, kp, lp )
				#A_tensor4 = apply_major_sym_to_tensor_term( A_tensor4, i, j, kp, lp )
				
			if (flaga is False) and (flagb is False):
			
				A_tensor4[i][j][k][l] = A_voigt4[a][b]
				#A_tensor4 = apply_minor_sym_to_tensor_term( A_tensor4, i, j, k, l )
				#A_tensor4 = apply_major_sym_to_tensor_term( A_tensor4, i, j, k, l )
			
			#print "A_voigt4[",a ,"][", b, "] = " , "A_tensor4[", i, "][", j, "][", k, "][", l, "]" 
			#print A_tensor4[i][j][k][l], A_voigt4[a][b]
			#print "-------------------------------------------------------------------"

	return A_tensor4
	
def apply_minor_sym_to_tensor_term( A_tensor4, i, j, k, l ):
	A_tensor4[j][i][k][l] = A_tensor4[i][j][k][l]
	A_tensor4[i][j][l][k] = A_tensor4[i][j][k][l]
	A_tensor4[j][i][l][k] = A_tensor4[i][j][k][l]
	return A_tensor4
						
def apply_major_sym_to_tensor_term( A_tensor4, i, j, k, l ):
	A_tensor4[k][l][i][j] = A_tensor4[i][j][k][l]
	return A_tensor4
	
	
#Only convenient if initial and final base given, else, determine P by hand, much faster
def generate_trans_matrix( init_base, final_base ):
	P = initTensor(0, 3, 3)
	for i in range(len(init_base)):
		for j in range(len(init_base)):
			#print "--------------------------"
			#print init_base[i], final_base[j]
			P[i][j] = vector_dot_vector( init_base[i], final_base[j] )
			#print i, j, P[i][j]
			
	return P

#Scalar product
def vector_dot_vector( vectora, vectorb ):
	sumdot = 0
	for i in range(len(vectora)):
		sumdot = sumdot + vectora[i] * vectorb[i]
		#print vectora[i], " * ", vectorb[i]
	#print vectora, " times ", vectorb, " = ", sumdot
	return sumdot

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

#! Square matrix
def matrix_dot_matrix( matrixa, matrixb ):
	C = initTensor( 0, 3, 3)
	
	for i in range( len(matrixc) ):
		for j in range( len(matrixc)):
			for k in range( len( matrixa )):
				C[i][j] = C[i][j] + matrixa[i][k]*matrix[k][j]
	return C


def matrix_dot_vector( matrixa, vector ):
	C = initTensor( 0, len(matrixa[0]), len(vector))
	if (len(matrixa[0]) != len(vector)):
	     print "This is not good... You're trying to perform a matrix dot vector with impossiburu dimensions!!"
	     
	for i in range( len(matrixa) ):
		for j in range( len(matrixa[0])):
			for k in range( len( vector )):
				C[i][j] = C[i][j] + matrixa[i][k]*vector[k]
	return C


def tensorial_base_change( P, tensorA ):
	tensorB = initTensor( 0, 3, 3, 3, 3)
	for m in range( len( tensorA[0][0][0] ) ):
		for n in range( len( tensorA[0][0][0] ) ):
			for o in range( len( tensorA[0][0][0] ) ):
				for p in range( len( tensorA[0][0][0] ) ):
					
					for i in range( len( tensorA[0][0][0] ) ):
						for j in range( len( tensorA[0][0][0] ) ):
							for k in range( len( tensorA[0][0][0] ) ):
								for l in range( len( tensorA[0][0][0] ) ):
									tensorB[m][n][o][p] = tensorB[m][n][o][p] + P[i][m]*P[j][n]*P[k][o]*P[l][p]*tensorA[i][j][k][l]
	return tensorB

	

def voigt_to_matrix( A_tensor2_voigtshape ):
	A_tensor2_matrix = initTensor(0, 3, 3)
	
	A_tensor2_matrix[0][0] = A_tensor2_voigtshape[0]
	A_tensor2_matrix[1][1] = A_tensor2_voigtshape[1]
	A_tensor2_matrix[2][2] = A_tensor2_voigtshape[2]
		
	A_tensor2_matrix[1][2] = A_tensor2_voigtshape[3]/sqrt(2)
	A_tensor2_matrix[2][1] = A_tensor2_voigtshape[3]/sqrt(2)
		
	A_tensor2_matrix[2][0] = A_tensor2_voigtshape[4]/sqrt(2)
	A_tensor2_matrix[0][2] = A_tensor2_voigtshape[4]/sqrt(2)
		
	A_tensor2_matrix[0][1] = A_tensor2_voigtshape[5]/sqrt(2)
	A_tensor2_matrix[1][0] = A_tensor2_voigtshape[5]/sqrt(2)
	
	return A_tensor2_matrix
	
	
def voigt_to_matrix( A_tensor2_matrix ):
	A_tensor2_voigt = initTensor(0, 6)
	
	A_tensor2_voigt[0] = A_tensor2_matrix[0][0]
	A_tensor2_voigt[1] = A_tensor2_matrix[1][1]
	A_tensor2_voigt[2] = A_tensor2_matrix[2][2]
	
	A_tensor2_voigt[3] = A_tensor2_matrix[1][2]*sqrt(2)
	A_tensor2_voigt[4] = A_tensor2_matrix[2][0]*sqrt(2)
	A_tensor2_voigt[5] = A_tensor2_matrix[0][1]*sqrt(2)
	
	return A_tensor2_voigt

#Square matrix, any size
def transpose_matrix( A_matrix ):
	A_transposed = initTensor( 0., len(A_matrix[0]), len(A_matrix[0]) )
	for i in range( 0, len(A_matrix[0])):
		for j in range( 0, len(A_matrix[0])):
			A_transposed[j][i] = A_matrix[i][j]
	return A_transposed


def tensor4_contract4_tensor4( A, B ):
	temp_sum = 0
	
	for i in range(0, len(A[0][0][0])):
		for j in range(0, len(A[0][0][0])):
			for k in range(0, len(A[0][0][0])):
				for l in range(0, len(A[0][0][0])):
					temp_sum = temp_sum + A[i][j][k][l]*B[i][j][k][l]
	return temp_sum


#Put an isotropic tensor in, get alpha and beta to build S_invert
def extract_isotropic_parameters( S_matrix ):
	
	S_tensor4 = voigt4_to_tensor4( S_matrix )
	
	J_tensor4 = generate_J_tensor4()
	
	I_tensor4 = generate_I_tensor4()
	
	K_tensor4 = generate_K_tensor4()
	
	alpha = tensor4_contract4_tensor4( J_tensor4 , S_tensor4 )
	print "Alpha =", alpha
	beta = tensor4_contract4_tensor4( K_tensor4, S_tensor4 )/5.
	print "beta =", beta
	
	print ""
	print "To get S_tensor4 invert:"
	print "S_invert = 1/alpha * J_tensor4 + 1/beta * K_tensor4"
	print "1/alpha =", 1./alpha
	print "1/beta =", 1./beta
	
	return alpha, beta
	
#Put an isotropic tensor in, get alpha and beta to build S_invert
def extract_cubic_parameters( S_matrix ):
	
	S_tensor4 = voigt4_to_tensor4( S_matrix )
	
	e1 = [ 1., 0., 0. ]
	e2 = [ 0., 1., 0. ]
	e3 = [ 0., 0., 1. ]
	
	Z_tensor4 = initTensor( 0., 3, 3, 3, 3 )
	
	for i in range(0, len(Z_tensor4[0][0][0])):
		for j in range(0, len(Z_tensor4[0][0][0])):
			for k in range(0, len(Z_tensor4[0][0][0])):
				for l in range(0, len(Z_tensor4[0][0][0])):
					Z_tensor4[i][j][k][l] = e1[i]*e1[j]*e1[k]*e1[l]+e2[i]*e2[j]*e2[k]*e2[l]+e3[i]*e3[j]*e3[k]*e3[l]

	J_tensor4 = generate_J_tensor4()
	
	I_tensor4 = generate_I_tensor4()
	
	K_tensor4 = generate_K_tensor4()
	
	KA_tensor4 = initTensor( 0., 3, 3, 3, 3 )
	for i in range( len( KA_tensor4[0][0][0] ) ):
		for j in range( len( KA_tensor4[0][0][0] ) ):
			for k in range( len( KA_tensor4[0][0][0] ) ):
				for l in range( len( KA_tensor4[0][0][0] ) ):
					KA_tensor4[i][j][k][l]= (Z_tensor4[i][j][k][l]-J_tensor4[i][j][k][l])
					
	KB_tensor4 = initTensor( 0., 3, 3, 3, 3 )
	for i in range( len( KB_tensor4[0][0][0] ) ):
		for j in range( len( KB_tensor4[0][0][0] ) ):
			for k in range( len( KB_tensor4[0][0][0] ) ):
				for l in range( len( KB_tensor4[0][0][0] ) ):
					KB_tensor4[i][j][k][l]= (I_tensor4[i][j][k][l]-Z_tensor4[i][j][k][l])
					
					
	alpha = tensor4_contract4_tensor4( J_tensor4, S_tensor4 )
	print "Alpha =", alpha
	beta = tensor4_contract4_tensor4( KA_tensor4, S_tensor4 )
	print "beta =", beta/2.
	gamma = tensor4_contract4_tensor4( KB_tensor4, S_tensor4 )
	print "gamma =", gamma/3.
	
	print ""
	print "S_inv = 1./alpha * J + 1./beta * Ka + 1./gamma * Kb "
	print "1/alpha = ", 1./alpha
	print "1/beta = ", 1./beta
	print "1/gamma = ", 1./gamma
	
	return alpha, beta, gamma
	
def generate_symmetric_matrix66_from_list( C ):

	C_matrix = [ [ C[0] , C[1] , C[2] , C[3] , C[4] , C[5] ],
	[	C[1] , C[6] , C[7] , C[8] , C[9] , C[10]],
	[	C[2] , C[7] , C[11], C[12], C[13], C[14]],
	[	C[3] , C[8] , C[12], C[15], C[16], C[17]],
	[	C[4] , C[9] , C[13], C[16], C[18], C[19]],
	[	C[5] , C[10], C[14], C[17], C[19], C[20]] ]
	
	return C_matrix