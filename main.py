# -*- coding: utf-8 -*-
"""
Created on Mon May 02 17:26:41 2016

@author: ilyass.tabiai@gmail.com
@author:
"""

from tensors import tensors
from csv_reader import csv_reader
from materials import materials
import pdb

# Let's import the data necessary for the problem
data = csv_reader()

tensor = tensors()

reinf = materials()
reinf.set_CNT()
pdb.set_trace()

