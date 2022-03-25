# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 12:33:44 2020

@author: yashr
"""

def task(n):
     for row in range(0,n):
         for column in range(0, row+1):
              print("* " , end="")
         print("\r")
 
task(6)

