from __future__ import print_function
import torch
from model import highwayNet
from utils import ngsimDataset,maskedMSETest
from torch.utils.data import DataLoader
import time
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


## Network Arguments
args = {}
args['use_cuda'] = True
args['encoder_size'] = 64
args['decoder_size'] = 128
args['in_length'] = 16
args['out_length'] = 5
args['grid_size'] = (13,3)


args['input_embedding_size'] = 32

args['train_flag'] = False


# Evaluation metric:

metric = 'rmse'

# Initialize network
net = highwayNet(args)
net.load_state_dict(torch.load('trained_models/sta_lstm_10272020.tar'))
if args['use_cuda']:
    net = net.cuda()

tsSet = ngsimDataset('TestSet.mat')
tsDataloader = DataLoader(tsSet,batch_size=1,shuffle=True,num_workers=0,collate_fn=tsSet.collate_fn) # 

lossVals = torch.zeros(5).cuda()
counts = torch.zeros(5).cuda()
lossVal = 0 # revised by Lei
count = 0

vehid = []
pred_x = []
pred_y = []
T = []
dsID = []
ts_cen = []
ts_nbr = []
wt_ha = []

hist_list = []
nbrs_list = []
gt_list = []

scene_num = 10
ct = 0
for i, data in enumerate(tsDataloader):
    # st_time = time.time()
    hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask, veh_id, t, ds = data
    ### hist shape: (his_length,batch_size, 2)
    ### nbrs shape: (his_length,nbr_batch_size, 2) : nbr_batch_size: #nbr1+#nbr2+...
    ### mask shape: (batch_size, 3, 13, encoder_size)

    if not isinstance(hist, list): # nbrs are not zeros
        vehid.append(veh_id) # current vehicle to predict

        T.append(t) # current timet
        dsID.append(ds)
    

    # Initialize Variables
        if args['use_cuda']:
            hist = hist.cuda()
            nbrs = nbrs.cuda()
            mask = mask.cuda()
            lat_enc = lat_enc.cuda()
            lon_enc = lon_enc.cuda()
            fut = fut.cuda()
            op_mask = op_mask.cuda()



        fut_pred, weight_ts_center, weight_ts_nbr, weight_ha= net(hist, nbrs, mask, lat_enc, lon_enc)
        # l, c = maskedMSETest(fut_pred, fut, op_mask)

        fut_pred_x = fut_pred[:,:,0].detach()
        fut_pred_x = fut_pred_x.cpu().numpy()

        fut_pred_y = fut_pred[:,:,1].detach()
        fut_pred_y = fut_pred_y.cpu().numpy()
        pred_x.append(fut_pred_x)
        pred_y.append(fut_pred_y)
        hist_list.append(hist.detach().cpu().numpy())
        nbrs_list.append(nbrs.detach().cpu().numpy())
        gt_list.append(fut.detach().cpu().numpy())

        # ts_cen.append(weight_ts_center[:, :, 0].detach().cpu().numpy())
        # ts_nbr.append(weight_ts_nbr[:, :, 0].detach().cpu().numpy())
        # wt_ha.append(weight_ha[:, :, 0].detach().cpu().numpy())



        # lossVal +=l.detach() # revised by Lei
        # count += c.detach()
        ct += 1
        if ct > scene_num:
            break
print(len(pred_x))

# Plot the x-y coordinates:

idx = 0
xbar = pred_x[idx]
ybar = pred_y[idx]
hist_target = hist_list[idx]
nbrs_target = nbrs_list[idx]
fut_target = gt_list[idx]

plt.plot(hist_target[:,0,0], hist_target[:,0,1],'r--',label='target history')
for i in range(nbrs_target.shape[1]):
    plt.plot(nbrs_target[:,i,0], nbrs_target[:,i,1],'g--',label='neighbor history')
plt.plot(xbar, ybar, color='red',label='target prediction')
plt.plot(fut_target[:,0,0], fut_target[:,0,1],'b--',label='target ground truth')
plt.show()