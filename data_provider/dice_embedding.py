#!/usr/bin/env python
# coding: utf-8

# In[9]:

from scipy.optimize import minimize_scalar

import pandas as pd 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import math
import numpy as np

import os

class DICE:
    '''
    DICE class turns numbers into their respective DICE embeddings
    
    Since the cosine function decreases monotonically between 0 and pi, simply employ a linear mapping
    to map distances s_n \\in [0, |a-b|] to angles \\theta \\in [0, pi]
    '''
    def __init__(self, d=2, min_bound=0, max_bound=100, norm="l2", save_path="sequence_learning/BPE"):
        
        assert d != 2, "RNN DICE embeddings have more numbers"
        self.d = d 

        self.min_bound = min_bound
        self.max_bound = max_bound
        self.norm = norm  # Restrict x and y to be of unit length

        save_path = save_path + "_%d_%d-%d.npz" % (d, min_bound, max_bound)

        if os.path.exists(save_path):

            arr = np.load(save_path)
            self.M, self.Q, self.R, self.Q_inv = arr['M'], arr['Q'], arr['R'], arr["Q_inv"]
            print ("Loaded DICE model from %s" % save_path)

        else:

            self.M = np.random.normal(0, 1, (self.d, self.d))
            self.Q, self.R = np.linalg.qr(self.M, mode="complete")  # QR decomposition for orthonormal basis, Q
                
            self.Q_inv = np.linalg.inv(self.Q)
        
            print ("Saved DICE model to %s" % save_path)
            np.savez(save_path, M=self.M, Q=self.Q, R=self.R, Q_inv=self.Q_inv)
            
        #self.all_numbers = np.array([self.make_dice(n) for n in range(min_bound, max_bound + 1)])
        self.all_numbers = self.make_dice(np.arange(min_bound, max_bound + 1))
    
    def generate_random_hypersphere_vector(self, N, theta):
        
        # Thanks Gemini
        # Vectorized code using prompting

        s = np.sin(theta)
        c = np.cos(theta)
        
        shape = (N,) + s.shape if len(s.shape) > 0 else N
        arr = np.empty(shape)
        arr[0,...] = 1.0
        arr[1:,...] = s
        np.cumprod(arr, out=arr, axis=0)  # Generates [1, s, s^2, ..., s^(N-1)] in-place
        arr[:-1,...] *= c             # Gives [c, s*c, s^2*c, ..., s^(N-2)*c, s^(N-1)]
        arr[-1,...] *= s
        return arr    

    def make_dice(self, num):
        
        norm_diff = np.array(num) / abs(self.max_bound - self.min_bound + 1) # added +1 for domain <=pi
        theta = norm_diff * np.pi
        
        #polar_coord = np.array([math.sin(theta)**(dim-1) * math.cos(theta) if dim < self.d \
        #                        else math.sin(theta)**(self.d) for dim in range(1, self.d+1)])
        polar_coord = self.generate_random_hypersphere_vector(self.d, theta)

        dice = np.dot(self.Q, polar_coord)  # DICE-D embedding for `num`
        
        # return dice.tolist()
        return dice

    def estimate_theta(self, k, sigma=None, margin=0.3, ngrid=4096, iters=60, bounded=False):
        
        # Thanks Claude

        k = np.atleast_2d(np.asarray(k, float)); B, M = k.shape
        w = 1.0 / np.broadcast_to(np.ones(()) if sigma is None
                                  else np.asarray(sigma, float), k.shape) ** 2
        lo, hi = (0.0, np.pi) if bounded else (-margin, np.pi + margin)
        g = lo + (np.arange(ngrid) + 0.5) * (hi - lo) / ngrid
        fg, _ = self.model(g, M)
        cost = w @ (fg**2).T - 2 * (w * k) @ fg.T + (w * k**2).sum(1, keepdims=True)
        t = g[np.argmin(cost, axis=1)]
        for _ in range(iters):                                   # Gauss-Newton polish
            f, df = self.model(t, M)
            step = (w * df * (f - k)).sum(1) / np.maximum((w * df**2).sum(1), 1e-300)
            t = t - step
            if bounded: t = np.clip(t, 0.0, np.pi)
            if np.all(np.abs(step) < 1e-15): break
        _, df = self.model(t, M)
        se = 1 / np.sqrt((w * df**2).sum(1))
        return (t[0], se[0]) if B == 1 else (t, se)

    def model(self, theta, M):
        """Model entries and d/dtheta. theta: (...,) -> (..., M)."""
        t = np.asarray(theta, float)[..., None]
        j = np.arange(M)
        s, c = np.sin(t), np.cos(t)
        cosfac = np.where(j < M - 1, 1.0, 0.0)
        spm1 = np.where(j > 0, s ** np.maximum(j - 1, 0), 0.0)
        f = s ** j * np.where(cosfac > 0, c, 1.0)
        df = np.where(cosfac > 0,
                      spm1 * (j * c**2 - s**2) + (j == 0) * (-s),
                      spm1 * j * c)
        return f, df

    def get_number(self, embedding, metric="mse"):
        
        # mse works best until gaussian error std = 10^-4
        assert metric in ["mse", "mae", "gauss-newton"]
        
        if len(embedding.shape) > 1:
            shape = (1,) * (len(embedding.shape) - 1) + (-1,)
            all_nums = self.all_numbers.view(shape)
        else:
            all_nums = self.all_numbers
            
        if metric == "mse":
            ret = (all_nums - embedding)**2
            #ret = (self.all_numbers - embedding)**2
            return self.min_bound + np.argmin(ret.sum(axis=-1))

        elif metric == "mae":
            ret = np.abs(all_nums - embedding)
            #ret = np.abs(self.all_numbers - embedding)
            return self.min_bound + np.argmin(ret.sum(axis=-1))

        else:
            #TODO: Vectorization

            embedding = self.Q_inv @ embedding
            theta, s = self.estimate_theta(embedding)

            num = theta * np.abs(self.max_bound - self.min_bound + 1) / math.pi
            return np.round(num).astype(np.int32)
