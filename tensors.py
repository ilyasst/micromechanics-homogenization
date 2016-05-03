# -*- coding: utf-8 -*-
"""
Created on Mon May 02 17:26:41 2016

@author: ilyass.tabiai@gmail.com
@author:
"""
import pdb
import math

class tensors():

	def __init__(self):
		print "Initialized functions from the tensor class"
		self.generate_I_tensor4()
		self.generate_J_tensor4()
		self.J_matrix = self.tensor4_to_voigt4( self.J_tensor )
		self.generate_K_tensor4()
		self.K_matrix = self.tensor4_to_voigt4( self.K_tensor )

	def initTensor(self, value, *lengths):
		"""
		Generate a tensor, any order, any size
		Value is the default value, commonly 0   
		"""
		list = []
		dim = len(lengths)
		if dim == 1:
			for i in range(lengths[0]):
				list.append(value)
		elif dim > 1:
			for i in range(lengths[0]):
				list.append(self.initTensor(value, *lengths[1:]))
		return list

	def tensor4_to_voigt4( self, A_tensor4 ):
		A_voigt4 = self.initTensor(0., 6, 6)
	
		#blue
		for i in range( 0, 3 ):
			for j in range( 0, 3 ):
				A_voigt4[i][j] = A_tensor4[i][i][j][j]
		

		for i in range( 0, 3 ):
			#print i
			A_voigt4[3][i] = math.sqrt(2) * A_tensor4[1][2][i][i]
			A_voigt4[4][i] = math.sqrt(2) * A_tensor4[2][0][i][i]
			A_voigt4[5][i] = math.sqrt(2) * A_tensor4[0][1][i][i]
		
		for j in range( 0, 3 ):
			#print j
			A_voigt4[j][3] = math.sqrt(2) * A_tensor4[j][j][1][2]
			A_voigt4[j][4] = math.sqrt(2) * A_tensor4[j][j][2][0]
			A_voigt4[j][5] = math.sqrt(2) * A_tensor4[j][j][0][1]
		
		A_voigt4[3][3] = 2 * A_tensor4[1][2][1][2]
		A_voigt4[4][3] = 2 * A_tensor4[2][0][1][2]
		A_voigt4[5][3] = 2 * A_tensor4[0][1][1][2]
	
		A_voigt4[3][4] = 2 * A_tensor4[1][2][2][0]
		A_voigt4[4][4] = 2 * A_tensor4[2][0][2][0]
		A_voigt4[5][4] = 2 * A_tensor4[0][1][2][0]

		A_voigt4[3][5] = 2 * A_tensor4[1][2][0][1]
		A_voigt4[4][5] = 2 * A_tensor4[2][0][0][1]
		A_voigt4[5][5] = 2 * A_tensor4[0][1][0][1]
	
		return A_voigt4
		
	def generate_I_tensor4(self):
		I_tensor4 = self.initTensor(0., 3, 3, 3, 3)
		for i in range( len( I_tensor4[0][0][0] ) ):
			for j in range( len( I_tensor4[0][0][0] ) ):
				for k in range( len( I_tensor4[0][0][0] ) ):
					for l in range( len( I_tensor4[0][0][0] ) ):
						I_tensor4[i][j][k][l]=(1./2.)*( self.kronecker(i,k)*self.kronecker(j,l)+self.kronecker(i,l)*self.kronecker(j,k) )
		self.I_tensor = I_tensor4

	def generate_J_tensor4(self):
		J_tensor4 = self.initTensor(0., 3, 3, 3, 3)
		for i in range( len( J_tensor4[0][0][0] ) ):
			for j in range( len( J_tensor4[0][0][0] ) ):
				for k in range( len( J_tensor4[0][0][0] ) ):
					for l in range( len( J_tensor4[0][0][0] ) ):
						J_tensor4[i][j][k][l]=(1./3.)*self.kronecker(i,j)*self.kronecker(k,l)
		self.J_tensor = J_tensor4

	def generate_K_tensor4(self):
		K_tensor4 = self.initTensor(0., 3, 3, 3, 3)
		for i in range( len( K_tensor4[0][0][0] ) ):
			for j in range( len( K_tensor4[0][0][0] ) ):
				for k in range( len( K_tensor4[0][0][0] ) ):
					for l in range( len( K_tensor4[0][0][0] ) ):
						K_tensor4[i][j][k][l]= ( self.I_tensor[i][j][k][l]-self.J_tensor[i][j][k][l] )		
		self.K_tensor = K_tensor4
		
	def kronecker( self, i, j ):
		if ( i == j ):
			return 1.
		else:
			return 0.

