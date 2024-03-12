# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 13:47:39 2024

@author: Antonina
"""

from ..config import tf, tfp
from .surrogate import Surrogate


class WindSurrogate(Surrogate):
    """Summation of the terms using phi angles """

    def __init__(self,  input_transform=None, output_transform=None):
        self._input_transform = input_transform
        self._output_transform = output_transform

    def __call__(self, inputs, var_list):
        return self.forward(inputs, var_list)

    def forward(self, inputs, var_list):
        # outputs=var_list[0]*inputs
        
        w = tf.linspace(0.1, 10.0, 1000)  # Example frequency range
        
        dw = w[1]-w[0]
        Lv=1.72
        S=6.868*w*Lv/((1+10.302*w*Lv)**(5/3))
        An = tf.sqrt(2 * S*dw)
        phi=var_list[0:len(w)]
        outputs = 100*tf.sqrt(2.0) * tf.reduce_sum(An * tf.cos(w*inputs +phi), axis=1)
        
        return inputs, output
    
    #var_list -phi angles
    # time=input 
    