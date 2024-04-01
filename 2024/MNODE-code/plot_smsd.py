import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import time
import matplotlib.colors as colors
from torch.utils.data import TensorDataset, DataLoader
from Model.model import *
from Model.utils import *
from Model.force_fun import *
from Model.Data_generator import *
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gd_smsd=np.load("saved_data/Single_Mass_Spring_Damper/Ground_truth for Single_Mass_Spring_Damper with training_size=300_num_steps_test=400_dt=0.01.npy")
pnode_smsd=np.load("saved_data/Single_Mass_Spring_Damper/PNODE for Single_Mass_Spring_Damper with training_size=300_num_steps_test=400_dt=0.01.npy")
lstm_smsd=np.load("saved_data/Single_Mass_Spring_Damper/LSTM for Single_Mass_Spring_Damper with training_size=300_num_steps_test=400_dt=0.01.npy")
fcnn_smsd=np.load("saved_data/Single_Mass_Spring_Damper/FCNN for Single_Mass_Spring_Damper with training_size=300_num_steps_test=400_dt=0.01.npy")

pnode_smsd=pnode_smsd[0:400]
lstm_smsd=lstm_smsd[0:400]
print("gd_smsd shape:",gd_smsd.shape)
print("pnode_smsd shape:",pnode_smsd.shape)
print("lstm_smsd shape:",lstm_smsd.shape)
print("fcnn_smsd shape:",fcnn_smsd.shape)

t_train= np.linspace(0, 0.01*300, 300)
t_test = np.linspace(0, 0.01*400, 400)
subplot_labels = ['(a): MNODE', '(b): MNODE', '(c): LSTM', '(d): LSTM', '(e): FCNN', '(f): FCNN']

label_fontsize = 15
legend_fontsize = 15
# Vertical position of the label below the x-axis
vertical_position = -0.22  # Adjust this as needed
fig,axs=plt.subplots(3,2,figsize=(10,10))
mse_pnode = np.mean((gd_smsd-pnode_smsd)**2)
mse_lstm = np.mean((gd_smsd-lstm_smsd)**2)
mse_fcnn = np.mean((gd_smsd-fcnn_smsd)**2)
axs[0, 0].plot(t_test, gd_smsd[:,:,0], 'b-', label="Ground truth")
axs[0, 0].plot(t_test[:300], pnode_smsd[:300,:,0], 'r--', label="Model training")
axs[0, 0].plot(t_test[300:], pnode_smsd[300:,:,0], 'r:', label="Model testing".format(mse_pnode))
axs[0, 0].grid()
axs[0, 0].set_xlabel(r"$t$",fontsize=label_fontsize)
axs[0, 0].set_ylabel(r"$x$",fontsize=label_fontsize)
axs[0, 0].text(0.5, vertical_position, subplot_labels[0], ha='center', va='top', fontsize=label_fontsize, transform=axs[0, 0].transAxes)

axs[0, 1].plot(t_test, gd_smsd[:,:,1], 'b-', label="Ground truth")
axs[0, 1].plot(t_test[:300], pnode_smsd[:300,:,1], 'r--', label="Model training")
axs[0, 1].plot(t_test[300:], pnode_smsd[300:,:,1], 'r:', label="Model testing ($\epsilon$={:.1e})".format(mse_pnode))
axs[0, 1].grid()
axs[0, 1].set_xlabel(r"$t$",fontsize=label_fontsize)
axs[0, 1].set_ylabel(r"$v$",fontsize=label_fontsize)
axs[0, 1].text(0.5, vertical_position, subplot_labels[1], ha='center', va='top', fontsize=label_fontsize, transform=axs[0, 1].transAxes)
axs[1, 0].plot(t_test, gd_smsd[:,:,0], 'b-', label="Ground truth")
axs[1, 0].plot(t_test[:300], lstm_smsd[:300,:,0], 'r--', label="Model training")
axs[1, 0].plot(t_test[300:], lstm_smsd[300:,:,0], 'r:', label="Model testing ($\epsilon$={:.1e})".format(mse_lstm))
axs[1, 0].grid()
axs[1, 0].set_xlabel(r"$t$",fontsize=label_fontsize)
axs[1, 0].set_ylabel(r"$x$",fontsize=label_fontsize)
axs[1, 0].text(0.5, vertical_position, subplot_labels[2], ha='center', va='top', fontsize=label_fontsize, transform=axs[1, 0].transAxes)

axs[1, 1].plot(t_test, gd_smsd[:,:,1], 'b-', label="Ground truth")
axs[1, 1].plot(t_test[:300], lstm_smsd[:300,:,1], 'r--', label="Model training")
axs[1, 1].plot(t_test[300:], lstm_smsd[300:,:,1], 'r:', label="Model testing ($\epsilon$={:.1e})".format(mse_lstm))
axs[1, 1].grid()
axs[1, 1].set_xlabel(r"$t$",fontsize=label_fontsize)
axs[1, 1].set_ylabel(r"$v$",fontsize=label_fontsize)
axs[1, 1].text(0.5, vertical_position, subplot_labels[3], ha='center', va='top', fontsize=label_fontsize, transform=axs[1, 1].transAxes)


axs[2, 0].plot(t_test, gd_smsd[:,:,0], 'b-', label="Ground truth")
axs[2, 0].plot(t_test[:300], fcnn_smsd[:300,:,0], 'r--', label="Model training")
axs[2, 0].plot(t_test[300:], fcnn_smsd[300:,:,0], 'r:', label="Model testing ($\epsilon$={:.1e})".format(mse_fcnn))
axs[2, 0].grid()
axs[2, 0].set_xlabel(r"$t$",fontsize=label_fontsize)
axs[2, 0].set_ylabel(r"$x$",fontsize=label_fontsize)
axs[2, 0].text(0.5, vertical_position, subplot_labels[4], ha='center', va='top', fontsize=label_fontsize, transform=axs[2, 0].transAxes)

axs[2, 1].plot(t_test, gd_smsd[:,:,1], 'b-', label="Ground truth")
axs[2, 1].plot(t_test[:300], fcnn_smsd[:300,:,1], 'r--', label="Model training")
axs[2, 1].plot(t_test[300:], fcnn_smsd[300:,:,1], 'r:', label="Model testing ($\epsilon$={:.1e})".format(mse_fcnn))
axs[2, 1].grid()
axs[2, 1].set_xlabel(r"$t$",fontsize=label_fontsize)
axs[2, 1].set_ylabel(r"$v$",fontsize=label_fontsize)
axs[2, 1].text(0.5, vertical_position, subplot_labels[5], ha='center', va='top', fontsize=label_fontsize, transform=axs[2, 1].transAxes)
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.99), ncol=3, fontsize=legend_fontsize)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("figures/Single_Mass_Spring_Damper/smsd_x_v_time.png")
plt.show()

subplot_labels =  ['(a): MNODE', '(b): LSTM', '(c): FCNN']
vertical_position = -0.18  # Adjust this as needed
fig2,axs2=plt.subplots(1,3,figsize=(10,4))
axs2[0].plot(gd_smsd[:,:,0],gd_smsd[:,:,1],label="Ground truth",color="blue",linestyle="-")
axs2[0].plot(pnode_smsd[:300,:,0],pnode_smsd[:300,:,1],label="Model training",color="red",linestyle="--")
# for the prediction, we plot in two steps first 300 steps for on the training range and the last 100 steps for testing
axs2[0].plot(pnode_smsd[300:,:,0],pnode_smsd[300:,:,1],label=r"Model testing",color="red",linestyle=":")
axs2[0].grid()
axs2[0].text(0.5, vertical_position, subplot_labels[0], ha='center', va='top', fontsize=label_fontsize, transform=axs2[0].transAxes)
#plt.title("Single Mass Spring Damper")
axs2[0].set_xlabel(r"$x$",fontsize=label_fontsize)
axs2[0].set_ylabel(r"$v$",fontsize=label_fontsize)
axs2[1].plot(gd_smsd[:,:,0],gd_smsd[:,:,1],label="Ground truth",color="blue",linestyle="-")
axs2[1].plot(lstm_smsd[:300,:,0],lstm_smsd[:300,:,1],label="Model training",color="red",linestyle="--")
# for the prediction, we plot in two steps first 300 steps for on the training range and the last 100 steps for testing
axs2[1].plot(lstm_smsd[300:,:,0],lstm_smsd[300:,:,1],label=r"Model testing",color="red",linestyle=":")
axs2[1].grid()
axs2[1].set_xlabel(r"$x$",fontsize=label_fontsize)
axs2[1].set_ylabel(r"$v$",fontsize=label_fontsize)
axs2[1].text(0.5, vertical_position, subplot_labels[1], ha='center', va='top', fontsize=label_fontsize, transform=axs2[1].transAxes)
axs2[2].plot(gd_smsd[:,:,0],gd_smsd[:,:,1],label="Ground truth",color="blue",linestyle="-")
axs2[2].plot(fcnn_smsd[:300,:,0],fcnn_smsd[:300,:,1],label="Model training",color="red",linestyle="--")
# for the prediction, we plot in two steps first 300 steps for on the training range and the last 100 steps for testing
axs2[2].plot(fcnn_smsd[300:,:,0],fcnn_smsd[300:,:,1],label=r"Model testing",color="red",linestyle=":")
axs2[2].grid()
axs2[2].set_xlabel(r"$x$",fontsize=label_fontsize)
axs2[2].set_ylabel(r"$v$",fontsize=label_fontsize)
axs2[2].text(0.5, vertical_position, subplot_labels[2], ha='center', va='top', fontsize=label_fontsize, transform=axs2[2].transAxes)
handles, labels = axs2[0].get_legend_handles_labels()
fig2.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.03), ncol=3, fontsize=legend_fontsize)
fig2.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("figures/Single_Mass_Spring_Damper/smsd_phase_space.png")
plt.show()
#print the mse for each model
mse_pnode = np.mean((gd_smsd-pnode_smsd)**2)
mse_lstm = np.mean((gd_smsd-lstm_smsd)**2)
mse_fcnn = np.mean((gd_smsd-fcnn_smsd)**2)
print("mse_pnode:",mse_pnode)
print("mse_lstm:",mse_lstm)
print("mse_fcnn:",mse_fcnn)
