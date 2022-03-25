# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 12:44:38 2020

@author: yashr
"""


def task(n):
      k = 2*n -2
      for i in range(n,-2,-2):
           for j in range(k,0,-1):
                print(end=" ")
           k = k +1
           for j in range(0, i+1):
                print("*", end=" ")
           print("\r")
task(8)