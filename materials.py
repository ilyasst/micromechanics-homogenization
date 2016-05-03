# -*- coding: utf-8 -*-
"""
Created on Mon May 02 17:26:41 2016

@author: ilyass.tabiai@gmail.com
@author:
"""

from tensors import tensors

class materials():

	def __init__(self):
		print "Material class initialized"
	
	def set_CNT(self):
		self.define_CNT_rigidity()
		
	def set_Epoxy(self):
		self.define_epoxy_properties()
		

	def define_CNT_rigidity(self):
		self.C = [ 40.7, 39.3, 12.4, 0., 0., 0., 40.7, 12.4, 0., 0., 0., 625.7, 0., 0., 0., 2.44, 0., 0., 2.44, 0., 1.36 ]

	def define_epoxy_properties(self):
		self.E = 2.
		self.nu = 0.3
		compute_bulk_shear(self)

	def define_volumic_ratio(self, ratio):
		self.v = ratio

	def compute_bulk_shear(self):
		self.k = self.E/(3.*(1. - 2.*self.nu)) 
		self.mu = E0/(2.*(1. + self.nu))

	def compute_epoxy_matrix(self):
		tensor= tensors()

		J_matrix = tensor4_to_voigt4( J_tensor4 )

		K_tensor4 = generate_K_tensor4()
		K_matrix = tensor4_to_voigt4( K_tensor4 )

		C0_tensor4 = dot( 3.*self.k, J_tensor4 ) + dot( 2.*self.mu, K_tensor4 )
		C0_matrix = tensor4_to_voigt4( C0_tensor4 )
