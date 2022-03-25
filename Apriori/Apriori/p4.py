# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 12:56:42 2020

@author: yashr
"""



def FibonacciSeries(n): 
      
    f1 = 0
    f2 = 1
    if (n < 1): 
        return
    for x in range(0, n): 
        print(f2, end = " ") 
        next = f1 + f2 
        f1 = f2 
        f2 = next
          
 
 
print(FibonacciSeries(10) )