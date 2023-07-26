#-*- coding : utf-8 -*-
# coding: utf-8

import time
import numpy as np
import torch
import h5py
import functools
import math
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from datetime import datetime
import parameters
from scipy import io
import sys
sys.path.append("..") 

print('current time:',datetime.now())

## GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

## import net
from Missing_trace import rdn as myNet
from Missing_trace import fixed_loss
net = myNet()

def SNR(noisy,gt):
    res = noisy - gt
    msegt = np.mean(gt * gt)
    mseres = np.mean(res * res)
    SNR = 10 * math.log((msegt/mseres),10)
    return SNR

## load data
sample_size_test = parameters.sample_size_test

Y = np.empty([sample_size_test,1,parameters.img_resolution1,parameters.img_resolution2])
X = np.empty([sample_size_test,1,parameters.img_resolution1,parameters.img_resolution2])  

f = h5py.File(parameters.test_data_path, 'r')
X[:,:,:,:] = f['X'][0:sample_size_test,:,:]
Y[:,:,:,:] = f['Y'][0:sample_size_test,:,:]
f.close()

## load test data in GPU
Xt = Variable(torch.from_numpy(X))
Xt = Xt.to(device)
Xt = Xt.type(torch.cuda.FloatTensor)

## test
net.load_state_dict(torch.load(parameters.result_path+str(parameters.test_checkpoint_epoch)+'.pkl'))

net = net.to(device)

with torch.no_grad():
    Y_hat = net(Xt)
    Y_hat = Y_hat.data.cpu().numpy()
    Y_hat = Y_hat.reshape(sample_size_test,parameters.img_resolution1,parameters.img_resolution2)

## plot
import matplotlib.pyplot as plt
extent = [0, 1, 1, 0]
plt.figure(figsize=(14,5))
temp = parameters.sample_id_test
cmax = np.max(dat)
cmin = -cmax
colour = 'gray'

plt.subplot(1,3,1)     
plt.title('X, SNR=%.4f'%(SNR(X[temp],Y[temp])),fontsize=18)   
plt.imshow(X[temp],vmax=cmax,vmin=cmin,extent=extent,cmap=colour)
plt.yticks(size=15)
plt.xticks(size=15)

plt.subplot(1,3,2)     
plt.title('y_hat, SNR=%.4f'%(SNR(y_hat[temp],Y[temp])),fontsize=18)
plt.imshow(dat_bp[temp],vmax=cmax,vmin=cmin,extent=extent,cmap=colour)
plt.yticks(size=15)
plt.xticks(size=15)

plt.subplot(1,3,3)     
plt.title('Y',fontsize=18)
plt.imshow(Y[temp],vmax=cmax,vmin=cmin,extent=extent,cmap=colour)
plt.yticks(size=15)
plt.xticks(size=15)

plt.tight_layout()
plt.savefig('')
plt.show()
