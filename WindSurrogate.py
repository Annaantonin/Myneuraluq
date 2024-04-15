# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 13:47:39 2024

@author: Antonina
"""

from ..config import tf, tfp
from .surrogate import Surrogate

# In this case, posterior, although named statisitically, is in fact a set of trainable variables, which are treated in 
#    the same way as in conventional machine learning problems.
class WindSurrogate(Surrogate):
    """Summation of the terms using phi angles """

    def __init__(self,  input_transform=None, output_transform=None):
        self._input_transform = input_transform
        self._output_transform = output_transform

    def __call__(self, inputs, var_list):
        # inputs-time
        # outputs-force generated with the spectra
        #var_list- phi angles
        return self.forward(inputs, var_list)

    def forward(self, inputs, var_list):
        # outputs=var_list[0]*inputs
        w = tf.linspace(0.01, 10.0, 1000)  # Frequency array
        dw = w[1]-w[0]
        phi=var_list
          
        Lv = 1.72 #  integral length scale parameter/ slowly-varying mean Uz
        
        S = 6.868 * w * Lv / ((1 + 10.302 * w * Lv) ** (5 / 3)) # Turbulence spectrum 
        Sw = (1.2 * 8 * 0.12 * Uz ** 2) ** 2 * S  # Power spectrum (air density*Area of the structure*drag coefficint)*spectrum of turbulent fluctuations
        
        cos_term = tf.math.cos(tf.tensordot(w, t[:,0], axes=0) + phi[:, tf.newaxis]) 
        
        srm_term = tf.sqrt(2 * Sw * dw)[:, tf.newaxis] * cos_term  # Broadcast sqrt_term over t
        
        f0 = tf.reduce_sum(srm_term, axis=0)  # Reduced turbulence part
          
        f1 = 0.5 * 1.2 * 8 *Uz ** 2  # w_mean part
        output = f0 + f1
        outputs=tf.reshape(output, [-1, 1]) 

        return inputs, outputs
    
    