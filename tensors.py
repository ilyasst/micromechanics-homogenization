# -*- coding: utf-8 -*-
"""
Created on Mon May 02 17:26:41 2016

@author: ilyass.tabiai@gmail.com
@author:
"""

class tensors():

    def __init__(self):
        print "Initialized functions from the tensor class"

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
