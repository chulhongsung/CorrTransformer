import os
os.chdir('/Users/chulhongsung/Desktop/lab/xml/proposal')

import numpy as np
import tensorflow as tf
from tensorflow import keras as K
from module import *

class CorrTransformer(K.models.Model):
    def __init__(self, num_labels, em_dim, dim_enc, dim_dec, m, d_model, num_heads):
        super(CorrTransformer, self).__init__()
        self.embedding_layer = tf.keras.layers.Embedding(num_labels+1, em_dim)
        self.enc = Encoder(dim_enc)
        self.dec = Decoder(dim_dec)
        self.set_transformer = SetTransformer(m, d_model, num_heads)
        
        
    def call(self, pair):
        x, y = pair
        emb_label = self.embedding_layer(y)
        hx = self.enc(x)
        gx = self.dec(hx)
        hy = self.set_transformer(emb_label)
    
        return hx, hy, gx