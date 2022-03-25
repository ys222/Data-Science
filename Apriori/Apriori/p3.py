# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 12:51:43 2020

@author: yashr
"""

  
def remove(input): 
  

    input = input.split(" ") 
  

    for i in range(0, len(input)): 
        input[i] = "".join(input[i]) 
    
    
if __name__ == "__main__": 
    input = 'My name is Yashraj varun and yours name is yashraj too but your surname is not varun'
    remove(input) 