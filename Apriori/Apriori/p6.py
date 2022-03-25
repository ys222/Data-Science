# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 13:26:18 2020

@author: yashr
"""


def sort(val): 
    return val[1]  
  

list = [(1, 2), (7, 10), (1, 1), (3, 10)] 
  
 
list.sort(key = sort)  
print(list) 