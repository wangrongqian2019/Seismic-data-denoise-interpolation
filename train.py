#!/usr/bin/env python
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
sample_size_train = parameters.sample_size_train
sample_size_test = parameters.sample_size_test

x = np.zeros([sample_size_train, 1,parameters.img_resolution1, parameters.img_resolution2])
y = np.zeros([sample_size_train, 1,parameters.img_resolution1, parameters.img_resolution2])

Y = np.empty([sample_size_test,1,parameters.img_resolution1,parameters.img_resolution2])
X = np.empty([sample_size_test,1,parameters.img_resolution1,parameters.img_resolution2])  

f = h5py.File(parameters.data_path, 'r')
x[:,:,:,:] = f['X'][0:sample_size_train,:,:]
y[:,:,:,:] = f['Y'][0:sample_size_train,:,:]
f.close()
 
f = h5py.File(parameters.test_data_path, 'r')
X[:,:,:,:] = f['X'][0:sample_size_test,:,:]
Y[:,:,:,:] = f['Y'][0:sample_size_test,:,:]
f.close()

## parameter
class MyDataset(Dataset):
    def __init__(self, a, b):
        self.data_1 = a
        self.data_2 = b  

    def __len__(self):
        return len(x)
    
    def __getitem__(self, idx):
        in_put = self.data_1[idx]
        out_put = self.data_2[idx]
        return in_put, out_put

batchsize = parameters.batchsize
dataset = MyDataset(x,y)
train_iter = DataLoader(dataset, batch_size=batchsize, shuffle=True, num_workers=10, drop_last=False, pin_memory=True)

lr = parameters.learning_rate
num_epochs = parameters.num_epochs

optimizer = torch.optim.Adam(net.parameters(), lr = lr, betas=(0.9, 0.999), eps=1e-8)

scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=parameters.end_lr)

loss_res = np.zeros(num_epochs) 
valida_res = np.zeros(num_epochs)

#gpu_tracker = MemTracker() 
#gpu_tracker.track()
#net=torch.nn.DataParallel(net)

if parameters.checkpoint_epoch>0:
    net.load_state_dict(torch.load(parameters.result_path+str(parameters.checkpoint_epoch)+'.pkl'))

net = net.to(device)
#gpu_tracker.track()

print("training on ", device)
loss = fixed_loss()#torch.nn.MSELoss(reduction='sum') #torch.nn.L1Loss(reduction='sum')

## load test data in GPU
Xt = Variable(torch.from_numpy(X))
Xt = Xt.to(device)
Xt = Xt.type(torch.cuda.FloatTensor)
#gpu_tracker.track()

for epoch in range(num_epochs):
    train_l_sum = 0.0
    start = time.time()
    batch_count = 0
    
    for xtrain, ytrain in train_iter:
        xtrain = xtrain.to(device).type(torch.cuda.FloatTensor)
        ytrain = ytrain.to(device).type(torch.cuda.FloatTensor)
        y_hat = net(xtrain)
        l = loss(torch.squeeze(y_hat), torch.squeeze(ytrain))
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        train_l_sum += l.cpu().item()
        batch_count += 1
    scheduler.step()

    with torch.no_grad():
        Y_hat = net(Xt)
        Y_hat = Y_hat.data.cpu().numpy()
        Y_hat = Y_hat.reshape(sample_size_test,1,parameters.img_resolution1,parameters.img_resolution2)
        #Y_hat = Y_hat/np.max(Y_hat)
        snr = np.mean(SNR(Y_hat,Y))
    print('epoch %d, loss %.6f, validation %.6f, time %.1f sec'
          % (epoch +parameters.checkpoint_epoch+ 1, train_l_sum/batch_count , snr, time.time() - start))

    loss_res[epoch] = train_l_sum/batch_count 
    valida_res[epoch] = snr
    if ((epoch+1) % 100) == 0:
        torch.save(net.state_dict(), parameters.result_path+str(epoch+parameters.checkpoint_epoch+1)+'.pkl')
        io.savemat(parameters.result_path+'training_epoch.mat',{'loss_res':loss_res,'valida_res':valida_res})

np.savetxt('loss.csv', loss_res, delimiter = '')
np.savetxt('validation.csv', valida_res, delimiter = '')

print('smallest error on trainging set:',np.argmin(loss_res)+1,'smallest error on test set:',np.argmax(valida_res)+1)
print('current time:',datetime.now())
