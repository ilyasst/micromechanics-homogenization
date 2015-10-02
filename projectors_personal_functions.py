from numpy import *
from tensor_personal_functions import *
from convenient_objects import *

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

def generate_I_tensor4():
	I_tensor4 = initTensor(0., 3, 3, 3, 3)
	for i in range( len( I_tensor4[0][0][0] ) ):
		for j in range( len( I_tensor4[0][0][0] ) ):
			for k in range( len( I_tensor4[0][0][0] ) ):
				for l in range( len( I_tensor4[0][0][0] ) ):
					I_tensor4[i][j][k][l]=(1./2.)*( kronecker(i,k)*kronecker(j,l)+kronecker(i,l)*kronecker(j,k) )
	return I_tensor4
	
def generate_J_tensor4():
	J_tensor4 = initTensor(0., 3, 3, 3, 3)
	for i in range( len( J_tensor4[0][0][0] ) ):
		for j in range( len( J_tensor4[0][0][0] ) ):
			for k in range( len( J_tensor4[0][0][0] ) ):
				for l in range( len( J_tensor4[0][0][0] ) ):
					J_tensor4[i][j][k][l]=(1./3.)*kronecker(i,j)*kronecker(k,l)
	return J_tensor4
	
def generate_K_tensor4():
	I_tensor4 = generate_I_tensor4()
	J_tensor4 = generate_J_tensor4()
	K_tensor4 = initTensor(0., 3, 3, 3, 3)
	for i in range( len( K_tensor4[0][0][0] ) ):
		for j in range( len( K_tensor4[0][0][0] ) ):
			for k in range( len( K_tensor4[0][0][0] ) ):
				for l in range( len( K_tensor4[0][0][0] ) ):
					K_tensor4[i][j][k][l]= ( I_tensor4[i][j][k][l]-J_tensor4[i][j][k][l] )		
	return K_tensor4

#Isotropic transverse
def generate_iT_matrix( axis ):
	print "====================================================================="
	print "Determining iT:"
	print "AXIS for transverse isotropy is", axis
	identity_matrix = initTensor(0., 3, 3)
	for i in range(0, len(identity_matrix)):
		for j in range(0, len(identity_matrix)):
			identity_matrix[i][j] = kronecker(i, j)
	
	n = initTensor(0., 3)
	for i in range(0, len(n)):
		if (i == axis):
			n[i] = 1.
	
	nXn = outer(n, n)
	
	iT = initTensor(0., 3, 3)
	print "Thus, iT="
	for i in range(0, len(identity_matrix)):
		for j in range(0, len(identity_matrix)):
			iT[i][j] = identity_matrix[i][j] - nXn[i][j]
	print iT
	return iT
	
#Isotropic transverse
def generate_EL_tensor( axis ):
	print "====================================================================="
	print "Determining EL:"
	print "AXIS for transverse isotropy is", axis
	n = initTensor(0., 3)
	for i in range(0, len(n)):
		if (i == axis):
			n[i] = 1.
	
	#EL = n X n X n X n
	EL = initTensor(0., 3, 3, 3, 3)
	for i in range( len( EL[0][0][0] ) ):
		for j in range( len( EL[0][0][0] ) ):
			for k in range( len( EL[0][0][0] ) ):
				for l in range( len( EL[0][0][0] ) ):
					EL[i][j][k][l]=n[i]*n[j]*n[k]*n[l]
					
	print "Thus, EL in voigt notations:"
	EL_voigt = tensor4_to_voigt4( EL )
	for i in range(0, len(EL_voigt)):
		print EL_voigt[i]
	return EL
		
def generate_JT_tensor( iT ):
	print "====================================================================="
	print "Determining JT:"

	#EL = n X n X n X n
	JT = initTensor(0., 3, 3, 3, 3)
	for i in range( len( JT[0][0][0] ) ):
		for j in range( len( JT[0][0][0] ) ):
			for k in range( len( JT[0][0][0] ) ):
				for l in range( len( JT[0][0][0] ) ):
					JT[i][j][k][l]=(1./2.)*iT[i][j]*iT[k][l]
		
	print "Thus, JT in voigt notations:"
	JT_voigt = tensor4_to_voigt4( JT )
	for i in range(0, len(JT_voigt)):
		print JT_voigt[i]
	return JT
		
def generate_IT_matrix( axis ):
	print "====================================================================="
	print "Determining IT:"
	print "AXIS for transverse isotropy is", axis
	
	IT = initTensor(0., 6, 6)
	for i in range(0, 3):
		for j in range(0, 3):
			if ( i == j) and (axis != i):
				IT[i][j] = 1.
	if (axis == 0):
		IT[3][3] = 1.
	if (axis == 1):
		IT[4][4] = 1.
	if (axis == 2):
		IT[5][5] = 1.
		
	print "Thus, IT, which is a matri(6X6):"
	for i in range(0, len(IT)):
		print IT[i]
		
	return IT
		
def generate_KE_tensor( axis, iT_matrix  ):
	print "====================================================================="
	print "Determining KE:"
	
	n = initTensor(0, 3)
	for i in range(0, len(n)):
		if (i == axis):
			n[i] = 1.

	KE = initTensor(0., 3, 3, 3, 3)
	for i in range( len( KE[0][0][0] ) ):
		for j in range( len( KE[0][0][0] ) ):
			for k in range( len( KE[0][0][0] ) ):
				for l in range( len( KE[0][0][0] ) ):
					KE[i][j][k][l] = (1./6.)*(2.*n[i]*n[j] - iT_matrix[i][j])*( 2.*n[k]*n[l]-iT_matrix[k][l] )
					
	print "Thus, KE in voigt notations:"
	KE_voigt = tensor4_to_voigt4( KE )
	for i in range(0, len(KE_voigt)):
		print KE_voigt[i]
		
	return KE
	
def generate_KT_tensor( IT_matrix, JT_tensor  ):
	print "====================================================================="
	print "Determining KT:"
	IT_tensor = voigt4_to_tensor4( IT_matrix )
	
	KT_tensor = initTensor(0., 3, 3, 3, 3)
	for i in range( len( KT_tensor[0][0][0] ) ):
		for j in range( len( KT_tensor[0][0][0] ) ):
			for k in range( len( KT_tensor[0][0][0] ) ):
				for l in range( len( KT_tensor[0][0][0] ) ):
					KT_tensor[i][j][k][l] = IT_tensor[i][j][k][l] - JT_tensor[i][j][k][l]
					
	print "Thus, KT in voigt notations:"
	KT_voigt = tensor4_to_voigt4( KT_tensor )
	for i in range(0, len(KT_voigt)):
		print KT_voigt[i]
		
	return KT_tensor
	
def generate_KL_tensor( KT, KE ):
	print "====================================================================="
	print "Determining KL:"
	K = generate_K_tensor4()
	
	KL_tensor = initTensor(0., 3, 3, 3, 3)
	for i in range( len( KL_tensor[0][0][0] ) ):
		for j in range( len( KL_tensor[0][0][0] ) ):
			for k in range( len( KL_tensor[0][0][0] ) ):
				for l in range( len( KL_tensor[0][0][0] ) ):
					KL_tensor[i][j][k][l] = K[i][j][k][l] - KT[i][j][k][l] - KE[i][j][k][l]
					
	print "Thus, KL in voigt notations:"
	KL_voigt = tensor4_to_voigt4( KL_tensor )
	for i in range(0, len(KL_voigt)):
		print KL_voigt[i]
		
	return KL_tensor
	
def generate_F_tensor( axis, iT_matrix):
	print "====================================================================="
	print "Determining F:"
	
	n = initTensor(0, 3)
	for i in range(0, len(n)):
		if (i == axis):
			n[i] = 1.
			
	F_tensor = initTensor(0, 3, 3, 3, 3)
	for i in range( len( F_tensor[0][0][0] ) ):
		for j in range( len( F_tensor[0][0][0] ) ):
			for k in range( len( F_tensor[0][0][0] ) ):
				for l in range( len( F_tensor[0][0][0] ) ):
					F_tensor[i][j][k][l]=sqrt(2)/2.*(iT_matrix[i][j]*n[k]*n[l]);
					
					
	print "Thus, F in voigt notations:"
	F_voigt = tensor4_to_voigt4( F_tensor )
	for i in range(0, len(F_voigt)):
		print F_voigt[i]
		
	return F_tensor