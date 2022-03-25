# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 12:40:52 2020

@author: yashr
"""

def task(n):
    for row in range(n, -1, -1):
         for column in range(0, row+1):
              print("* " , end="")
         print("\r")
 
task(5)