#%%
import os
os.chdir('/Users/chulhongsung/Desktop/lab/xml/proposal')
#%%
import numpy as np
import tensorflow as tf
from tensorflow import keras as K
from module import *
#%%
class proposed_model(K.models.Model):
    def __init__(self, dim_enc, dim_dec, m, d_model, num_heads, k=1):
        self.enc = Encoder(dim_enc)
        self.dec = Decoder(dim_dec)
        self.set_transformer = SetTransformer(m, d_model, num_heads, k)
        
        
    def call(self, pair):
        x, y = pair
        hx = self.enc(x)
        gx = self.dec(hx)
        hy = self.set_transformer(y)
    
        return hx, hy, gx