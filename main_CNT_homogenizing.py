from import_data import *
from numpy import *
from tensor_personal_functions import *
from hmgnzt_personal_functions import *


def moyenne_orientation( C, angle0pi, angle02pi, N ):
	print
	print "Starting moyenne orientation..."
	print
	mx = 16*pow( 2, (N-1) )
	print "N=", N
	print "Max:", mx
	Pm = initTensor( 0., 3, 3 )
	#C_tensor4_moy = initTensor( 0., 3, 3, 3, 3)
	C_matrix_moy = initTensor( 0., 6, 6 )
	print 

	
	for theta in range( 0, mx ):
		for phi in range( 0, mx ):
			for beta in range( 0, mx ):
				
				Pm[0][0] = cos(angle0pi[N-1][theta][0])*cos(angle02pi[N-1][phi][0])*cos(angle02pi[N-1][beta][0])-sin(angle02pi[N-1][phi][0])*sin(angle02pi[N-1][beta][0])
				Pm[0][1] = -cos(angle0pi[N-1][theta][0])*cos(angle02pi[N-1][phi][0])*sin(angle02pi[N-1][beta][0])-sin(angle02pi[N-1][phi][0])*cos(angle02pi[N-1][beta][0])
				Pm[0][2] = sin(angle0pi[N-1][theta][0])*cos(angle02pi[N-1][phi][0])
				
				Pm[1][0] = cos(angle0pi[N-1][theta][0]) * sin(angle02pi[N-1][phi][0]) * cos(angle02pi[N-1][beta][0])  + cos(angle02pi[N-1][phi][0]) * sin(angle02pi[N-1][beta][0])
				Pm[1][1] = -cos(angle0pi[N-1][theta][0])*sin(angle02pi[N-1][phi][0])*sin(angle02pi[N-1][beta][0]) + cos(angle02pi[N-1][phi][0])*cos(angle02pi[N-1][beta][0])
				Pm[1][2] = sin(angle0pi[N-1][theta][0])*sin(angle02pi[N-1][phi][0])
				
				Pm[2][0] = - sin(angle0pi[N-1][theta][0]) * cos(angle02pi[N-1][beta][0])
				Pm[2][1] = sin(angle0pi[N-1][theta][0])*sin(angle02pi[N-1][beta][0])
				Pm[2][2] = cos(angle0pi[N-1][theta][0])
				
				#for m in range(0, 3):
					#for o in range(0, 3):
						#if Pm[m][o] < 0.00001:
							#Pm[m][o] = 0.
				
	
				for i in range(0, 3):
					for j in range(0, 3):
						for k in range(0, 3):
							for l in range(0, 3):
								#for m in range(0, 3):
									#for n in range(0, 3):
										#for o in range(0, 3):
											#for p in range(0, 3):
												#C_tensor4_moy[m][n][o][p] = C_tensor4_moy[m][n][o][p] + ( (1./(8.*pow( pi,2)))*Pm[i][m]*Pm[j][n]*Pm[k][o]*Pm[l][p]*C[i][j][k][l]*sin(angle0pi[N][theta][0])*angle0pi[N][theta][1]*angle02pi[N][phi][1]*angle02pi[N][beta][1])
								C_matrix_moy[0][0] = C_matrix_moy[0][0] + ( (1./(8.*pow( pi,2)))*Pm[i][0]*Pm[j][0]*Pm[k][0]*Pm[l][0]*C[i][j][k][l]*sin(angle0pi[N-1][theta][0])*angle0pi[N-1][theta][1]*angle02pi[N-1][phi][1]*angle02pi[N-1][beta][1])
								C_matrix_moy[1][1] = C_matrix_moy[1][1] + ( (1./(8.*pow( pi,2)))*Pm[i][1]*Pm[j][1]*Pm[k][1]*Pm[l][1]*C[i][j][k][l]*sin(angle0pi[N-1][theta][0])*angle0pi[N-1][theta][1]*angle02pi[N-1][phi][1]*angle02pi[N-1][beta][1])
								C_matrix_moy[2][2] = C_matrix_moy[2][2] + ( (1./(8.*pow( pi,2)))*Pm[i][2]*Pm[j][2]*Pm[k][2]*Pm[l][2]*C[i][j][k][l]*sin(angle0pi[N-1][theta][0])*angle0pi[N-1][theta][1]*angle02pi[N-1][phi][1]*angle02pi[N-1][beta][1])
								C_matrix_moy[3][3] = C_matrix_moy[3][3] + 2.*(1./(8.*pow( pi,2)))*Pm[i][1]*Pm[j][2]*Pm[k][1]*Pm[l][2]*C[i][j][k][l]*sin(angle0pi[N-1][theta][0])*angle0pi[N-1][theta][1]*angle02pi[N-1][phi][1]*angle02pi[N-1][beta][1]
								C_matrix_moy[4][4] = C_matrix_moy[4][4] + 2.*(1./(8.*pow( pi,2)))*Pm[i][2]*Pm[j][0]*Pm[k][2]*Pm[l][0]*C[i][j][k][l]*sin(angle0pi[N-1][theta][0])*angle0pi[N-1][theta][1]*angle02pi[N-1][phi][1]*angle02pi[N-1][beta][1]
								C_matrix_moy[5][5] = C_matrix_moy[5][5] + 2.*(1./(8.*pow( pi,2)))*Pm[i][0]*Pm[j][1]*Pm[k][0]*Pm[l][1]*C[i][j][k][l]*sin(angle0pi[N-1][theta][0])*angle0pi[N-1][theta][1]*angle02pi[N-1][phi][1]*angle02pi[N-1][beta][1]
								C_matrix_moy[0][1] = C_matrix_moy[0][1] + ( (1./(8.*pow( pi,2)))*Pm[i][0]*Pm[j][0]*Pm[k][1]*Pm[l][1]*C[i][j][k][l]*sin(angle0pi[N-1][theta][0])*angle0pi[N-1][theta][1]*angle02pi[N-1][phi][1]*angle02pi[N-1][beta][1])
								C_matrix_moy[0][2] = C_matrix_moy[0][2] + ( (1./(8.*pow( pi,2)))*Pm[i][0]*Pm[j][0]*Pm[k][2]*Pm[l][2]*C[i][j][k][l]*sin(angle0pi[N-1][theta][0])*angle0pi[N-1][theta][1]*angle02pi[N-1][phi][1]*angle02pi[N-1][beta][1])
								C_matrix_moy[1][2] = C_matrix_moy[1][2] + ( (1./(8.*pow( pi,2)))*Pm[i][1]*Pm[j][1]*Pm[k][2]*Pm[l][2]*C[i][j][k][l]*sin(angle0pi[N-1][theta][0])*angle0pi[N-1][theta][1]*angle02pi[N-1][phi][1]*angle02pi[N-1][beta][1])
	
	#C_matrix_moy = tensor4_to_voigt4( C_tensor4_moy )
	for i in range(0, 6):
		for j in range((i+1), 6):
			C_matrix_moy[j][i] = C_matrix_moy[i][j]
			
	return C_matrix_moy

#======================================================================================================

def temp_07_04():
	print
	print "Importing angles data..."

	zeta_csv = import_hmgnzt_quad( "zeta3_8points.csv" )
	omega_csv = import_hmgnzt_quad( "omega_512points.csv" )

	angle_0pi = []
	angle_0pi.append( import_anglepi( "0pi_16points.csv" ) )
	angle_0pi.append( import_anglepi( "0pi_32points.csv" ) )
	angle_0pi.append( import_anglepi( "0pi_64points.csv" ) )
	angle_0pi.append( import_anglepi( "0pi_128points.csv" ) )
	angle_0pi.append( import_anglepi( "0pi_256points.csv" ) )
	angle_0pi.append( import_anglepi( "0pi_512points.csv" ) )

	angle_2pi = initTensor( 0., 6 )
	angle_2pi[0] = import_anglepi( "02pi_16points.csv" )
	angle_2pi[1] = import_anglepi( "02pi_32points.csv" )
	angle_2pi[2] = import_anglepi( "02pi_64points.csv" )
	angle_2pi[3] = import_anglepi( "02pi_128points.csv" )
	angle_2pi[4] = import_anglepi( "02pi_256points.csv" )
	angle_2pi[5] = import_anglepi( "02pi_512points.csv" )
	
	E0 = 2.
	nu0 = 0.3
	E1 = 65.
	nu1 = 0.35
	
	v0 = 0.73 
	v1 = 1 - v0 

	k0 = E0/(3*(1 - 2*nu0)) 
	mu0 = E0/(2*(1 + nu0)) 
	k1 = E1/(3*(1 - 2*nu1))
	mu1 = E1/(2*(1 + nu1)) 
	
	J_tensor4 = generate_J_tensor4()
	J_matrix = tensor4_to_voigt4( J_tensor4 )

	K_tensor4 = generate_K_tensor4()
	K_matrix = tensor4_to_voigt4( K_tensor4 )

	C0_tensor4 = dot( 3.*k0, J_tensor4 ) + dot( 2.*mu0, K_tensor4 ) 
	C0_matrix = tensor4_to_voigt4( C0_tensor4 )
	C1_tensor4 = dot( 3.*k1, J_tensor4 ) + dot( 2.*mu1, K_tensor4 ) 
	C1_matrix = tensor4_to_voigt4( C1_tensor4 )
	
	a = [ 1., 1., 37.]
	
	#C_MT_matrix_ale = mori_tanaka( a, v0, C0_tensor4, C1_tensor4, zeta_csv, omega_csv )
	
	S_eshelby_quad_tensor4 = generate_eshelby_tensor( a, C0_tensor4, zeta_csv, omega_csv )
	S_eshelby_quad_tensor4 = clean_S_eshelby(S_eshelby_quad_tensor4)
	S_eshelby_quad_matrix = tensor4_to_voigt4( S_eshelby_quad_tensor4 )

	Iden_voigt4 = tensor4_to_voigt4( generate_I_tensor4() )
	C_difference = initTensor( 0., 6, 6 )
	for i in range(0, len(C0_matrix)):
		for j in range(0, len(C0_matrix)):
			C_difference[i][j] = C1_matrix[i][j] - C0_matrix[i][j]
	T0_matrix = Iden_voigt4
	T1_matrix = inv( Iden_voigt4 + dot( S_eshelby_quad_matrix, dot( inv(C0_matrix), (C_difference) ) ) )
	
	A0_matrix = inv( dot(v0,T0_matrix) + dot(v1,T1_matrix) )
	A1_matrix = dot( T1_matrix, A0_matrix )
	
	C_MT_matrix = dot( v1 ,dot( C_difference, A1_matrix ) )
	
	C_matrix_ale = moyenne_orientation( C_MT_matrix, angle_0pi, angle_2pi, N )

	C_tensor4_ale = voigt4_to_tensor4( C_matrix_ale )

	k_ale= tensor4_contract4_tensor4( J_tensor4, C_tensor4_ale )/3.
	mu_ale = tensor4_contract4_tensor4( K_tensor4, C_tensor4_ale )/10.

	print
	print "C_MT_matrix_ale:"
	for i in range(0, len(C_matrix_ale)):
		print C_matrix_ale[i]

	print 
	print "k_ale=", k_ale
	print "mu_ale=", mu_ale
	



	print 
	print "---------------------------------------------------------------------"
	print
	print "NTC alignes"

	C_MT_matrix_ali = mori_tanaka( a, v0, C0_tensor4, C_NTC_tensor4_t, zeta_csv, omega_csv )
	C_MT_tensor4_ali = voigt4_to_tensor4( C_MT_matrix_ali )
	k_ali = tensor4_contract4_tensor4( J_tensor4, C_MT_tensor4_ali )/3.
	mu_ali = tensor4_contract4_tensor4( K_tensor4, C_MT_tensor4_ali )/10.

	print
	print "C_MT_matrix_alignes:"
	for i in range(0, len(C_MT_matrix_ali)):
		print C_MT_matrix_ali[i]

	print 
	print "k_ali=", k_ali
	print "mu_ali=", mu_ali
	
	
	
	
	
	
	
	
	
	
#======================================================================================================

print
print "Importing angles data and qaudrature data for numerical integrations..."


zeta_csv = import_hmgnzt_quad( "zeta3_4points.csv" )
omega_csv = import_hmgnzt_quad( "omega_512points.csv" )

angle_0pi = []
angle_0pi.append( import_anglepi( "0pi_16points.csv" ) )
angle_0pi.append( import_anglepi( "0pi_32points.csv" ) )
angle_0pi.append( import_anglepi( "0pi_64points.csv" ) )
angle_0pi.append( import_anglepi( "0pi_128points.csv" ) )
angle_0pi.append( import_anglepi( "0pi_256points.csv" ) )
angle_0pi.append( import_anglepi( "0pi_512points.csv" ) )
   
angle_2pi = initTensor( 0., 6 )
angle_2pi[0] = import_anglepi( "02pi_16points.csv" )
angle_2pi[1] = import_anglepi( "02pi_32points.csv" )
angle_2pi[2] = import_anglepi( "02pi_64points.csv" )
angle_2pi[3] = import_anglepi( "02pi_128points.csv" )
angle_2pi[4] = import_anglepi( "02pi_256points.csv" )
angle_2pi[5] = import_anglepi( "02pi_512points.csv" )



C = [ 40.7, 39.3, 12.4, 0., 0., 0., 40.7, 12.4, 0., 0., 0., 625.7, 0., 0., 0., 2.44, 0., 0., 2.44, 0., 1.36 ]
C_NTC_matrix = generate_symmetric_matrix66_from_list( C )
print 
print "Rigidity matrix of carbon nanotubes:"
print "C_NTC_matrix="
for i in range( 0, len(C_NTC_matrix)):
	print C_NTC_matrix[i]
C_NTC_tensor4 = voigt4_to_tensor4(C_NTC_matrix)
print
wait = raw_input("PRESS SMTNG TO CONTINUE.")

print "----------------------------------------------------------------------"
print "Hypothesis 1: NTCs tend to aggregate as spheres. They are rigidly attached one to another."
print "----------------------------------------------------------------------"

N = 1

print " Now going to compute the new rigidity matrix C_NTC_matrix_tilde when the NTCs are gathered as isotropic spheres"

C_NTC_matrix_t = moyenne_orientation( C_NTC_tensor4, angle_0pi, angle_2pi, N )
print
print "C_NTC_matrix_tilde rigidity of NTCs when they stick together as spherical aggregates:"
for i in range(0, len(C_NTC_matrix_t)):
	print C_NTC_matrix_t[i]
C_NTC_tensor4_t = voigt4_to_tensor4(C_NTC_matrix_t)
print
wait = raw_input("PRESS SMTNG TO CONTINUE.")


print 
print "---------------------------------------------------------------------"
print "Now going to compute the situation where 5% of CNTs are homogeneously scattered in the matrix as spherical aggregates"
print "5% NTCs aggregated with Mori-Tanaka"

print
print "Getting Epoxy matrix properties..."
E0 = 2.
nu0 = 0.3

v0 = 0.95
v1 = 1. - v0

a = [ 1., 1., 1. ]

J_tensor4 = generate_J_tensor4()
J_matrix = tensor4_to_voigt4( J_tensor4 )

K_tensor4 = generate_K_tensor4()
K_matrix = tensor4_to_voigt4( K_tensor4 )

k0 = E0/(3.*(1. - 2.*nu0)) 
mu0 = E0/(2.*(1. + nu0)) 
C0_tensor4 = dot( 3.*k0, J_tensor4 ) + dot( 2.*mu0, K_tensor4 )
C0_matrix = tensor4_to_voigt4( C0_tensor4 )

print
print "C0_matrix rigidity of Epoxy properties:"
for i in range(0, len(C0_matrix)):
	print C0_matrix[i]
	
print "Let's now Mori-Tanaka the Epoxy and 5% Spherical aggregates of NTCs..."
print
wait = raw_input("PRESS SMTNG TO CONTINUE.")

print 

C_MT_matrix_aggregat = mori_tanaka( a, v0, C0_tensor4, C_NTC_tensor4_t, zeta_csv, omega_csv )
C_MT_tensor4_aggregat = voigt4_to_tensor4( C_MT_matrix_aggregat )
print "Done."
print
print "With 5% NTCs as spherical aggregates, the homogenized properties are given by:"
print "C_MT_matrix_aggregat:"
for i in range(0, len(C_MT_matrix_aggregat)):
	print C_MT_matrix_aggregat[i]


k_agg = tensor4_contract4_tensor4( J_tensor4, C_MT_tensor4_aggregat )/3.
mu_agg = tensor4_contract4_tensor4( K_tensor4, C_MT_tensor4_aggregat )/10.

print 
print "k_agg=", k_agg
print "mu_agg=", mu_agg
print
wait = raw_input("PRESS SMTNG TO CONTINUE.")

print 
print "---------------------------------------------------------------------"
print "Hypothesis 2: Now the NTCs are separated and randomly distributed in the matrix"
print "---------------------------------------------------------------------"

print
print "Epoxy properties, C0_matrix:"
for i in range(0, len(C0_matrix)):
	print C0_matrix[i]
print
print "Carbon Nanotubes' properties, C_NTC_matrix:"
for i in range(0, len(C_NTC_matrix)):
	print C_NTC_matrix[i]
	
print
wait = raw_input("PRESS SMTNG TO CONTINUE.")

print
print "Geometrical shape of NTCs given by:"
a = [ 1., 1., 500.]
print a

S_eshelby_quad_tensor4 = generate_eshelby_tensor( a, C0_tensor4, zeta_csv, omega_csv )
S_eshelby_quad_tensor4 = clean_S_eshelby(S_eshelby_quad_tensor4)
S_eshelby_quad_matrix = tensor4_to_voigt4( S_eshelby_quad_tensor4 )

print
print "S_eshelby_quad_matrix:"
for i in range(0, len(S_eshelby_quad_matrix)):
	print S_eshelby_quad_matrix[i]

C0_matrix = tensor4_to_voigt4( C0_tensor4 )

Im = tensor4_to_voigt4( generate_I_tensor4() )
C_difference = initTensor( 0., 6, 6 )
for i in range(0, len(C0_matrix)):
	for j in range(0, len(C0_matrix)):
		C_difference[i][j] = C_NTC_matrix[i][j] - C0_matrix[i][j]
T0_matrix = Im
T1_matrix = inv( Im + dot( S_eshelby_quad_matrix, dot( inv(C0_matrix), (C_difference) ) ) )

A0_matrix = inv( dot(v0,T0_matrix) + dot(v1,T1_matrix) )
A1_matrix = dot( T1_matrix, A0_matrix )
	
C_MT_matrix_ale = dot( v1 ,dot( C_difference, A1_matrix ) )
C_MT_tensor4_ale = voigt4_to_tensor4( C_MT_matrix_ale )

N = 1
C_matrix_ale = moyenne_orientation( C_MT_tensor4_ale, angle_0pi, angle_2pi, N )
C_tensor4_ale = voigt4_to_tensor4( C_matrix_ale )

k_ale= tensor4_contract4_tensor4( J_tensor4, C_tensor4_ale )/3.
mu_ale = tensor4_contract4_tensor4( K_tensor4, C_tensor4_ale )/10.


print
print "C_MT_matrix_ale:"
for i in range(0, len(C_MT_matrix_ale)):
	print C_MT_matrix_ale[i]
	
print
print "C_matrix_ale:"
for i in range(0, len(C_matrix_ale)):
	print C_matrix_ale[i]

print 
print "k_ale=", k_ale
print "mu_ale=", mu_ale

print 
print "---------------------------------------------------------------------"
print "Hypothesis 3"
print "NTC aligned in the same direction (x_3)"
print "---------------------------------------------------------------------"
print
wait = raw_input("PRESS SMTNG TO CONTINUE.")
#C_MT_matrix_ali = mori_tanaka( a, v0, C0_tensor4, C_NTC_tensor4_t, zeta_csv, omega_csv )

C_difference = initTensor( 0., 6, 6 )
for i in range(0, len(C0_matrix)):
	for j in range(0, len(C0_matrix)):
		C_difference[i][j] = C_NTC_matrix[i][j] - C0_matrix[i][j]
		
C_MT_matrix_ali = C0_matrix + dot( v1 ,dot( C_difference, A1_matrix ) )
C_MT_tensor4_ali = voigt4_to_tensor4( C_MT_matrix_ali )
k_ali = tensor4_contract4_tensor4( J_tensor4, C_MT_tensor4_ali )/3.
mu_ali = tensor4_contract4_tensor4( K_tensor4, C_MT_tensor4_ali )/10.

print
print "C_MT_matrix_aligned:"
for i in range(0, len(C_MT_matrix_ali)):
	print C_MT_matrix_ali[i]

print 
print "k_aligned=", k_ali
print "mu_aligned=", mu_ali

print 
print "---------------------------------------------------------------------"
print

S_MT_matrix_aggregat = inv( C_MT_matrix_aggregat)
E_aggregat = 1./S_MT_matrix_aggregat[0][0]
print 
print "E_aggregat=", E_aggregat
