# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 10:29:30 2024

@author: Antonina
"""

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

from tensorflow.python.client import device_lib 
print(device_lib.list_local_devices())
#%%
import Myneuraluq.neuraluq as neuq
import Myneuraluq.neuraluq.variables as neuq_vars
from Myneuraluq.neuraluq.config import tf

import numpy as np
import scipy.io as sio
from scipy.io import loadmat
import matplotlib.pyplot as plt


#%%
def load_data(noise):
    data = sio.loadmat("Data/sdof_tower.mat")
    x_tt_train, t_train = data["a"], data["t"]
    x_train, f_train = data["u"], data["pf"]
    
    n=4000
    return  x_tt_train[0:n], t_train[0:n], x_train[0:n], f_train[0:n]
#%%
class Surrogate:
    """Base class for all surrogate modules."""

    def __init__(self):
        self._input_transform = None
        self._output_transform = None

    def __call__(self):
        raise NotImplementedError("__call__ is not implemented.")

    @property
    def input_transform(self):
        return self._input_transform

    @property
    def output_transform(self):
        return self._output_transform


class WindSurrogate(Surrogate):
    """Summation of the terms using phi angles """

    def __init__(self,  input_transform=None, output_transform=None):
        self._input_transform = input_transform
        self._output_transform = output_transform

    def __call__(self, inputs, var_list):
        return self.forward(inputs, var_list)

    def forward(self, inputs, var_list):
        t=inputs
        w = tf.linspace(0.01, 10.0, 1000)  # Frequency array
        dw = w[1]-w[0]
        Uz=var_list[0]
        phi=var_list[1:len(w)]
          
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


def reduction_transform(output_tensor, output_size=1000):
    """
    Apply a dimensionality reduction transformation to the neural network output.
    
    Args:
    - output_tensor: The output tensor from the neural network.
    - output_size: The desired size of the output after dimensionality reduction.
    
    Returns:
    - The transformed output tensor with reduced dimensionality.
    """
    # Weight matrix for dimensionality reduction
    weights = tf.Variable(tf.random.normal([output_tensor.shape[-1], output_size]), dtype=tf.float32)
    # Bias vector
    bias = tf.Variable(tf.zeros([output_size]), dtype=tf.float32)
    
    # Apply linear transformation
    transformed_output = tf.matmul(output_tensor, weights)
    return transformed_output

#%%

def pde_init(t,  x): 
    x_t = tf.gradients(x, t)[0]
    return tf.concat([x[0], x_t[0]], axis=-1)

def pde_xtt(t,x,*args):
    x_t = tf.gradients(x, t)                                                                                                                                                                                                                        
    x_tt = tf.gradients(x_t, t)
    return x_tt

def pde_fn(t, x, f, log_c, log_k, *angles):
    m=4500
    x_t = tf.gradients(x, t)      [0] [..., 0:1]     #velocity                                                                                                                                                                                                                   
    x_tt=f-(0.245*tf.exp(log_c)*x_t + 27000*tf.exp(log_k)*x)
    return x_tt/m
    
def pde_Spec(t, f, phi, *angles):
    w = tf.linspace(0.01, 10.0, 1000)  # Frequency array, example values
    # phi=phi[0:w.shape[0]]
    # phi = tf.random.uniform(shape=[w.shape[0]], minval=0, maxval=2*np.pi, dtype=tf.float32)
    phi=tf.concat(angles, 0)
    print(phi)
    # Uz=1.37/0.4*np.log(30/2) 
    Uz=9.27  # Consider known for this example
    
    dw = w[1] - w[0]  # Frequency spacing
    Lv=1.72#  integral length scale parameter/ slowly-varying mean Uz
    S=6.868*w*Lv/((1+10.302*w*Lv)**(5/3)) # Compute Sw tensor
    Sw=(1.2*8*0.12*Uz**2)**2*S  
       
    cos_term = tf.math.cos(tf.tensordot(w, t[:,0], axes=0) + phi[:, tf.newaxis])  # tf.tensordot will broadcast w over t
    print(cos_term)
    srm_term = tf.sqrt(2 * Sw * dw)[:, tf.newaxis] * cos_term  # broadcast sqrt_term over t
    print(srm_term)
    # Sum over the frequencies, which is the first dimension (axis=0) after broadcasting
    f0 = tf.reduce_sum(srm_term, axis=0)  
    f1=0.5*1.2*8*Uz**2 #w_ mean part
    output=f0+f1
    return output-f


def WindSurrogate(inputs, var_list):   
    t=inputs
    w = tf.linspace(0.01, 10.0, 1000)  # Frequency array, example values
    # phi = tf.random.uniform(shape=[w.shape[0]], minval=0, maxval=2*np.pi, dtype=tf.float32)
    dw = w[1] - w[0]  # Frequency spacing
    Lv = 1.72
    print(var_list)
    Uz=var_list[0]
    phi=var_list[1:len(w)]
   
    S = 6.868 * w * Lv / ((1 + 10.302 * w * Lv) ** (5 / 3))
    Sw = (1.2 * 8 * 0.12 * Uz ** 2) ** 2 * S  # Compute Sw tensor
    
    cos_term = tf.math.cos(tf.tensordot(w, t[:,0], axes=0) + phi[:, tf.newaxis]) 
    
    srm_term = tf.sqrt(2 * Sw * dw)[:, tf.newaxis] * cos_term  # Broadcast sqrt_term over t
    
    f0 = tf.reduce_sum(srm_term, axis=0)  # Reduced turbulence part
      
    f1 = 0.5 * 1.2 * 8 *Uz ** 2  # w_mean part
    output = f0 + f1
    outputs=tf.reshape(output, [-1, 1]) 
    return outputs 

   
@neuq.utils.timer
def Trainable(
    x_tt_train, t_train, x_train, f_train, noise, layers, layers_phi
):
       
    process_x = neuq.process.Process(
        surrogate=neuq.surrogates.FNN(layers=layers, activation= tf.math.sin),
        posterior=neuq_vars.fnn.Trainable(layers=layers),# displacement
      )   
           
    # process_f = neuq.process.Process(
    #     surrogate=neuq.surrogates.FNN(layers=layers),
    #     posterior=neuq_vars.fnn.Trainable(layers=layers), # force
    # ) 
    process_f = neuq.process.Process(
        surrogate=WindSurrogate, 
        posterior=neuq_vars.fnn.Trainable(layers=layer_phi),   
    )    
  
    process_log_c = neuq.process.Process(
        surrogate=neuq.surrogates.Identity(),
        posterior=neuq_vars.const.Trainable(value=0.1),
    )
    process_log_k = neuq.process.Process(
        surrogate=neuq.surrogates.Identity(),
        posterior=neuq_vars.const.Trainable(value=0.1),
    )
    
    phi= neuq.process.Process(
        surrogate=neuq.surrogates.Identity(),
        posterior=neuq_vars.const.Trainable(value=10),
    )
    
    # phi=[]
    for i in range (1000):  
        new=neuq.process.Process(
            surrogate=neuq.surrogates.Identity(),
            posterior=neuq_vars.const.Trainable(value=2*np.pi),       )
        # phi[i]=phi
    phi.append(new)
    
    method = neuq.inferences.DEns(
        num_samples=1, num_iterations=20000, optimizer=tf.train.AdamOptimizer(1e-3),
    )
    
             
    loss_init = neuq.likelihoods.MSE(
        inputs=t_train[0:1],
        targets=[0,0], 
        processes=[process_x], 
        pde=pde_init,
        multiplier=1,
    )
    loss_x = neuq.likelihoods.MSE(   
        inputs=t_train,
        targets=x_tt_train,
        processes=[process_x],
        pde=pde_xtt, 
        multiplier=1,
    )    
    
    loss_f = neuq.likelihoods.MSE(
        inputs=t_train,
        targets=x_tt_train, 
        # targets=np.zeros_like(t_train), # minimizing the loss to be close to zero
        processes=[process_x, process_f,  process_log_c, process_log_k, phi], # tf train
        pde=pde_fn,
        multiplier=1,
    )
    
    loss_force = neuq.likelihoods.MSE(   #Loss xtt
        inputs=t_train , # or frequencies array
        targets=np.zeros_like(t_train),
        processes=[process_f, phi],
        pde=pde_Spec, 
        multiplier=1,
    )   
    
    
    likelihoods=[loss_init, 
                 loss_x, 
                 loss_f, 
                 loss_force]
    processes=[process_x, process_f, process_log_c, process_log_k, phi]
     # processes=[process_f]
 
    # build model
    model = neuq.models.Model(
        processes=[process_x, process_f, process_log_c, process_log_k,  phi],
        likelihoods=[loss_init, loss_x, loss_f, loss_force],
    )
    model.compile(method)

    samples = model.run()
    processes=processes 
    likelihoods=likelihoods
    # likelihoods=[]
    return processes, samples, model, likelihoods


if __name__ == "__main__":

    noise = 0

    x_tt_train, t_train, x_train, f_train = load_data(noise)

    layers = [1, 20, 20,  1]
    layer_phi=[1,100,1]

    processes, samples, model, likelihoods = Trainable( x_tt_train, t_train, x_train, f_train, noise, layers, layer_phi  )

    x_pred, f_pred, logc_pred, logk_pred, phi_pred= model.predict(t_train, samples, processes, pde_fn=None)
    
plt.plot(t_train,np.mean(x_pred, axis=(0, 2)),label='Predicted displacement')
plt.plot(t_train, x_train,'r',label='Actual displacement')
plt.legend()
#%%
x_pred, f_pred, logc_pred, logk_pred = model.predict(t_train, samples, processes, pde_fn=pde_fn)
import seaborn as sns
sns.set_theme()
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = [10, 6]
# (f_pred,) = model.predict(t_train, samples, processes, pde_fn=pde_fn)
plt.plot(t_train, f_pred[0,:,0], label='Predicted force')
# plt.plot(t_train, f_train, label='True force')
plt.xlabel('Time(s)')
plt.ylabel('Force (N)')
plt.legend()

plt.legend(loc='upper center', ncol=2)
# plt.savefig('SDOF force known')
#%%
(f_pred, )=model.predict(t_train, samples, processes, pde_fn=pde_Spec)
plt.plot(t_train, np.mean(f_pred, axis=(0, 2)), label='Predicted force')
plt.plot(t_train, f_train, label='True force')

#%%
(xtt_pred,) = model.predict(t_train, samples, processes, pde_fn=pde_fn)
plt.rcParams['figure.figsize'] = [10, 6]
plt.plot(t_train, xtt_pred[0,:,0], 'b', linewidth=3.5, label='Predicted acceleration')
plt.plot(t_train, x_tt_train,'r', label='Measured acceleration',alpha=0.85)
plt.legend(loc='upper center', ncol=2)
plt.xlabel('Time(s)')
plt.ylabel('Acceleration $ (m/s^2)$')

# plt.savefig('SDOF acceleration known')
#%%
(xtt_pred,) = model.predict(t_train, samples, processes, pde_fn=pde_xtt)
plt.rcParams['figure.figsize'] = [10, 7]
plt.plot(xtt_pred[0].flatten())
plt.plot(x_tt_train,'r')
#%%
plt.rcParams['figure.figsize'] = [5,5]
neuq.utils.hist(np.exp(logk_pred).flatten(), bins=30, name="value of $k$")
neuq.utils.hist(np.exp(logc_pred).flatten(), bins=30,  name="value of $c$")
neuq.utils.hist(Uz_pred.flatten(), bins=30,  name="value of $U(z)$")
#%%
f_mean = np.mean(800*f_pred, axis=0)
f_std = np.std(800*f_pred, axis=0)
plt.plot(t_train, f_train, label='True force')
plt.plot(t_train, f_mean, "r--", label="Predicted force")

# plt.plot(t_train, f_mean-2*f_std, "r--", label="mean")
# plt.plot(t_train, f_mean+2*f_std, "r--", label="mean")

plt.fill_between( t_train.flatten(),
                 f_mean.flatten() - 2 * np.sqrt(f_std.flatten()**2), 
                 f_mean.flatten() + 2 * np.sqrt(f_std.flatten()**2), alpha=0.3,
facecolor="r",
label="2 $\\bar{\\sigma}$")
