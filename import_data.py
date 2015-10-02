# -*- coding: utf-8 -*-
import csv
import os
from tensor_personal_functions import initTensor

def import_data_04_01( file_name ):
	i = 0
	time = []
	deformation = []
	
	with open( file_name, 'rb') as csvfile:
		spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
		for row in spamreader:
			if i == 0:
				i = i + 1
				pass
			else:
				i = i + 1
				time.append(float(row[0]))
				deformation.append( float(row[1]) )

	return time, deformation
	
def import_data_04_02( file_name ):
	i = 0
	time = []
	deformation11 = []
	deformation22 = []
	
	with open( file_name, 'rb') as csvfile:
		spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
		for row in spamreader:
			if i == 0:
				i = i + 1
				pass
			else:
				i = i + 1
				time.append(float(row[0]))
				deformation11.append( float(row[1]) )
				deformation22.append( float(row[2]) )

	return time, deformation11, deformation22


def import_data_05_04( file_name ):
	i = 0
	time = []
	
	with open( file_name, 'rb') as csvfile:
		spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
		total_lines = 0
		for line in spamreader:
			total_lines = total_lines + 1
		stress = initTensor(0, total_lines, 6)
		print "The file contains", total_lines, "lines."
		
		csvfile.seek(0)
		for row in spamreader:
			if i == 0:
				i = i + 1
				pass
			else:
				time.append( float(row[0]) )
				for j in range(0, 6):
					stress[i-1][j] = float(row[j+1])
				i = i + 1		
	return time, stress

def import_data_lab_fluage( file_name ):
	i = 0
	time = []
	
	stress = []
	
	with open( file_name, 'rb') as csvfile:
		spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
		total_lines = 0
		for line in spamreader:
			total_lines = total_lines + 1 
		strain = initTensor(0, total_lines-1, 6)
		
		print 
		print "Importing data from", file_name
		print "The file contains", total_lines, "lines."
		csvfile.seek(0)
		for row in spamreader:
			if i == 0:
				i = i + 1
				pass
			else:
				time.append( float( row[0]) )
				stress.append( float( row[1] ) )
				for j in range(0, 2):
					strain[i-1][j] = float( row[j+2] )
				i = i + 1	

	return time, stress, strain

def import_data_verif_eprouvette_simple( file_name ):
	i = 0
	time = []
	
	with open( file_name, 'rb') as csvfile:
		spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
		total_lines = 0
		for line in spamreader:
			total_lines = total_lines + 1 
		stress = initTensor(0, total_lines-1, 6)
		
		print 
		print "Importing data from", file_name
		print "The file contains", total_lines, "lines."
		csvfile.seek(0)
		for row in spamreader:
			if i == 0:
				i = i + 1
				pass
			else:
				time.append( float( row[0]) )
				stress[i-1][1] = float( row[1] )
				i = i + 1
	return time, stress

def import_data_verif_eprouvette_simple_ansys( file_name ):
	i = 0
	time = []
	
	with open( file_name, 'rb') as csvfile:
		spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
		total_lines = 0
		for line in spamreader:
			total_lines = total_lines + 1 
		strain = initTensor(0, total_lines-1, 6)
		
		print 
		print "Importing data from", file_name
		print "The file contains", total_lines, "lines."
		csvfile.seek(0)
		for row in spamreader:
			if i == 0:
				i = i + 1
				pass
			else:
				time.append( float( row[0]) )
				strain[i-1][1] = float( row[2] )
				strain[i-1][0] = float( row[1] )
				strain[i-1][3] = float( row[3] )
				i = i + 1
	return time, strain


def import_hmgnzt_quad( file_name ):
	i = 0
	
	with open( os.getcwd() + "/hmgnzt_data/" + file_name, 'rb' ) as csvfile:
		spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
		total_lines = 0
		for line in spamreader:
			total_lines = total_lines + 1 
		
		quad = initTensor(0., total_lines, 2)
		print
		print "Importing data from", file_name
		print "The file contains", total_lines, "lines."
		csvfile.seek(0)
		i = 0
		for row in spamreader:
			quad[i][0] = float(row[0])
			quad[i][1] = float(row[1])
			i = i + 1
	return quad

def import_anglepi( file_name ):
	i = 0
	
	with open( os.getcwd() + "/hmgnzt_data/" + file_name, 'rb' ) as csvfile:
		spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
		total_lines = 0
		for line in spamreader:
			total_lines = total_lines + 1 
			
		angle_pi = initTensor( 0., total_lines, 2 )
		print
		print "Importing data from", file_name
		print "The file contains", total_lines, "lines."
		csvfile.seek(0)
		i = 0
		for row in spamreader:
			angle_pi[i][0] = float(row[0])
			angle_pi[i][1] = float(row[1])
			i = i + 1
		
	return angle_pi
		
		
		