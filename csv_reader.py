# -*- coding: utf-8 -*-
"""
Created on Mon May 02 17:26:41 2016

@author: ilyass.tabiai@gmail.com
@author:
"""

import csv
import os
from tensors import tensors

class csv_reader():
    """
    This class reads data from csv files. It imports the quadrature
    values and angles for orientation averaging from csv files in
    the data folder.
    """
    def __init__(self):
        #self.zeta = self.import_hmgnzt_quad( "zeta3_4points.csv" )
        #self.omega = self.import_hmgnzt_quad( "zeta3_4points.csv" )
        self.angle_0pi = self.get_angle_0pi()
        self.angle_2pi = self.get_angle_2pi()        

    def import_hmgnzt_quad( self, file_name ):
	    i = 0
	    tensor = tensors()
	    with open( os.getcwd() + "/hmgnzt_data/" + file_name, 'rb' ) as csvfile:
		    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
		    total_lines = 0
		    for line in spamreader:
			    total_lines = total_lines + 1 
		
		    quad = tensor.initTensor(0., total_lines, 2)
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
        
    def get_angle_0pi(self):
        angle_0pi = []
        angle_0pi.append( self.import_anglepi( "0pi_16points.csv" ) )
        angle_0pi.append( self.import_anglepi( "0pi_32points.csv" ) )
        angle_0pi.append( self.import_anglepi( "0pi_64points.csv" ) )
        angle_0pi.append( self.import_anglepi( "0pi_128points.csv" ))
        angle_0pi.append( self.import_anglepi( "0pi_256points.csv") )
        angle_0pi.append( self.import_anglepi("0pi_512points.csv"))  
        return angle_0pi  
    
    def get_angle_2pi(self):
        tensor = tensors()
        angle_2pi = tensor.initTensor( 0., 6 )
        angle_2pi[0] = self.import_anglepi( "02pi_16points.csv" )
        angle_2pi[1] = self.import_anglepi( "02pi_32points.csv" )
        angle_2pi[2] = self.import_anglepi( "02pi_64points.csv" )
        angle_2pi[3] = self.import_anglepi( "02pi_128points.csv" )
        angle_2pi[4] = self.import_anglepi( "02pi_256points.csv" )
        angle_2pi[5] = self.import_anglepi( "02pi_512points.csv" )
        return angle_2pi
  
    def import_anglepi( self, file_name ):
	    i = 0
	    tensor = tensors()
	
	    with open( os.getcwd() + "/hmgnzt_data/" + file_name, 'rb' ) as csvfile:
		    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
		    total_lines = 0
		    for line in spamreader:
			    total_lines = total_lines + 1 
			
		    angle_pi = tensor.initTensor( 0., total_lines, 2 )
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
		

