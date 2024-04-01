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

subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

# Vertical position of the label below the x-axis
vertical_position = -0.17  # Adjust this as needed
fig,axs=plt.subplots(3,2,figsize=(10,10))
mse_pnode = np.mean((gd_smsd-pnode_smsd)**2)
axs[0, 0].plot(t_test, gd_smsd[:,:,0], 'b-', label="Ground truth")
axs[0, 0].plot(t_test[:300], pnode_smsd[:300,:,0], 'r--', label="Model training")
axs[0, 0].plot(t_test[300:], pnode_smsd[300:,:,0], 'r:', label="Model testing".format(mse_pnode))
axs[0, 0].grid()
axs[0, 0].set_xlabel(r"$t$")
axs[0, 0].set_ylabel(r"$x$")
axs[0, 0].text(0.5, vertical_position, subplot_labels[0], ha='center', va='top', fontsize=10, transform=axs[0, 0].transAxes)

axs[0, 1].plot(t_test, gd_smsd[:,:,1], 'b-', label="Ground truth")
axs[0, 1].plot(t_test[:300], pnode_smsd[:300,:,1], 'r--', label="Model training")
axs[0, 1].plot(t_test[300:], pnode_smsd[300:,:,1], 'r:', label="Model testing ($\epsilon$={:.1e})".format(mse_pnode))
axs[0, 1].grid()
axs[0, 1].set_xlabel(r"$t$")
axs[0, 1].set_ylabel(r"$v$")
axs[0, 1].text(0.5, vertical_position, subplot_labels[1], ha='center', va='top', fontsize=10, transform=axs[0, 1].transAxes)

mse_lstm = np.mean((gd_smsd-lstm_smsd)**2)
axs[1, 0].plot(t_test, gd_smsd[:,:,0], 'b-', label="Ground truth")
axs[1, 0].plot(t_test[:300], lstm_smsd[:300,:,0], 'r--', label="Model training")
axs[1, 0].plot(t_test[300:], lstm_smsd[300:,:,0], 'r:', label="Model testing ($\epsilon$={:.1e})".format(mse_lstm))
axs[1, 0].grid()
axs[1, 0].set_xlabel(r"$t$")
axs[1, 0].set_ylabel(r"$x$")
axs[1, 0].text(0.5, vertical_position, subplot_labels[2], ha='center', va='top', fontsize=10, transform=axs[1, 0].transAxes)

axs[1, 1].plot(t_test, gd_smsd[:,:,1], 'b-', label="Ground truth")
axs[1, 1].plot(t_test[:300], lstm_smsd[:300,:,1], 'r--', label="Model training")
axs[1, 1].plot(t_test[300:], lstm_smsd[300:,:,1], 'r:', label="Model testing ($\epsilon$={:.1e})".format(mse_lstm))
axs[1, 1].grid()
axs[1, 1].set_xlabel(r"$t$")
axs[1, 1].set_ylabel(r"$v$")
axs[1, 1].text(0.5, vertical_position, subplot_labels[3], ha='center', va='top', fontsize=10, transform=axs[1, 1].transAxes)

mse_fcnn = np.mean((gd_smsd-fcnn_smsd)**2)
axs[2, 0].plot(t_test, gd_smsd[:,:,0], 'b-', label="Ground truth")
axs[2, 0].plot(t_test[:300], fcnn_smsd[:300,:,0], 'r--', label="Model training")
axs[2, 0].plot(t_test[300:], fcnn_smsd[300:,:,0], 'r:', label="Model testing ($\epsilon$={:.1e})".format(mse_fcnn))
axs[2, 0].grid()
axs[2, 0].set_xlabel(r"$t$")
axs[2, 0].set_ylabel(r"$x$")
axs[2, 0].text(0.5, vertical_position, subplot_labels[4], ha='center', va='top', fontsize=10, transform=axs[2, 0].transAxes)

axs[2, 1].plot(t_test, gd_smsd[:,:,1], 'b-', label="Ground truth")
axs[2, 1].plot(t_test[:300], fcnn_smsd[:300,:,1], 'r--', label="Model training")
axs[2, 1].plot(t_test[300:], fcnn_smsd[300:,:,1], 'r:', label="Model testing ($\epsilon$={:.1e})".format(mse_fcnn))
axs[2, 1].grid()
axs[2, 1].set_xlabel(r"$t$")
axs[2, 1].set_ylabel(r"$v$")
axs[2, 1].text(0.5, vertical_position, subplot_labels[5], ha='center', va='top', fontsize=10, transform=axs[2, 1].transAxes)
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.97), ncol=3)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
#plt.savefig("figures/Single_Mass_Spring_Damper/smsd_x_v_time.png")
plt.show()



plt.subplot(3,2,2)
plt.plot(t_test,gd_smsd[:,:,1],label="Ground truth",color="blue",linestyle="-")
plt.plot(t_test[:300:],pnode_smsd[:300,:,1],label="Model training",color="red",linestyle="--")
# for the prediction, we plot in two steps first 300 steps for on the training range and the last 100 steps for testing
plt.plot(t_test[300:],pnode_smsd[300:,:,1],label=r"Model testing($\epsilon$={:.1e})".format(mse_pnode),color="red",linestyle=":")
#plt.legend(loc="upper right",frameon=False)
plt.grid()
plt.text(0.5, vertical_position, subplot_labels[1], ha='center', va='top', fontsize=10, transform=plt.gca().transAxes)
plt.xlabel(r"$t$")
plt.ylabel(r"$v$")
plt.subplot(3,2,3)
mse_lstm = np.mean((gd_smsd-lstm_smsd)**2)
plt.plot(t_test,gd_smsd[:,:,0],label="Ground truth",color="blue",linestyle="-")
plt.plot(t_test[:300:],lstm_smsd[:300,:,0],label="Model training",color="red",linestyle="--")
# for the prediction, we plot in two steps first 300 steps for on the training range and the last 100 steps for testing
plt.plot(t_test[300:],lstm_smsd[300:,:,0],label=r"Model testing($\epsilon$={:.1e})".format(mse_lstm),color="red",linestyle=":")
#plt.legend(loc="upper right",frameon=False)
plt.grid()
plt.xlabel(r"$t$")
plt.ylabel(r"$x$")
plt.text(0.5, vertical_position, subplot_labels[2], ha='center', va='top', fontsize=10, transform=plt.gca().transAxes)
plt.subplot(3,2,4)
plt.plot(t_test,gd_smsd[:,:,1],label="Ground truth",color="blue",linestyle="-")
plt.plot(t_test[:300:],lstm_smsd[:300,:,1],label="Model training",color="red",linestyle="--")
# for the prediction, we plot in two steps first 300 steps for on the training range and the last 100 steps for testing
plt.plot(t_test[300:],lstm_smsd[300:,:,1],label=r"Model testing($\epsilon$={:.1e})".format(mse_lstm),color="red",linestyle=":")
#plt.legend(loc="upper right",frameon=False)
plt.grid()
plt.text(0.5, vertical_position, subplot_labels[3], ha='center', va='top', fontsize=10, transform=plt.gca().transAxes)
plt.xlabel(r"$t$")
plt.ylabel(r"$v$")
plt.subplot(3,2,5)
mse_fcnn = np.mean((gd_smsd-fcnn_smsd)**2)
plt.plot(t_test,gd_smsd[:,:,0],label="Ground truth",color="blue",linestyle="-")
plt.plot(t_test[:300:],fcnn_smsd[:300,:,0],label="Model training",color="red",linestyle="--")
# for the prediction, we plot in two steps first 300 steps for on the training range and the last 100 steps for testing
plt.plot(t_test[300:],fcnn_smsd[300:,:,0],label=r"Model testing($\epsilon$={:.1e})".format(mse_fcnn),color="red",linestyle=":")
#plt.legend(loc="upper right",frameon=False)
plt.text(0.5, vertical_position, subplot_labels[4], ha='center', va='top', fontsize=10, transform=plt.gca().transAxes)
plt.grid()
plt.xlabel(r"$t$")
plt.ylabel(r"$x$")
plt.subplot(3,2,6)
plt.plot(t_test,gd_smsd[:,:,1],label="Ground truth",color="blue",linestyle="-")
plt.plot(t_test[:300:],fcnn_smsd[:300,:,1],label="Model training",color="red",linestyle="--")
# for the prediction, we plot in two steps first 300 steps for on the training range and the last 100 steps for testing
plt.plot(t_test[300:],fcnn_smsd[300:,:,1],label=r"Model testing($\epsilon$={:.1e})".format(mse_fcnn),color="red",linestyle=":")
#plt.legend(loc="upper right",frameon=False)
plt.grid()
plt.text(0.5, vertical_position, subplot_labels[5], ha='center', va='top', fontsize=10, transform=plt.gca().transAxes)
plt.xlabel(r"$t$")
plt.ylabel(r"$v$")
plt.tight_layout()
#plt.savefig("figures/Single_Mass_Spring_Damper/smsd_x_v_time.png")
plt.show()


#plot the phase space for the single mass spring damper
# make a plot with multiple subplots
# scientific notation!!
# scientific style!!
# Define the labels for each subplot
subplot_labels = ['(a)', '(b)', '(c)',]
vertical_position = -0.15  # Adjust this as needed
plt.subplots(1,3,figsize=(13,4))
plt.subplot(1,3,1)
# the legend should contrain the mean square error
mse_pnode = np.mean((gd_smsd-pnode_smsd)**2)
plt.plot(gd_smsd[:,:,0],gd_smsd[:,:,1],label="Ground truth",color="blue",linestyle="-")
plt.plot(pnode_smsd[:300,:,0],pnode_smsd[:300,:,1],label="Model training",color="red",linestyle="--")
# for the prediction, we plot in two steps first 300 steps for on the training range and the last 100 steps for testing
plt.plot(pnode_smsd[300:,:,0],pnode_smsd[300:,:,1],label=r"Model testing($\epsilon$={:.1e})".format(mse_pnode),color="red",linestyle=":")
plt.legend(loc="upper right",frameon=False)
plt.grid()
plt.text(0.5, vertical_position, subplot_labels[0], ha='center', va='top', fontsize=10, transform=plt.gca().transAxes)
#plt.title("Single Mass Spring Damper")
plt.xlabel(r"$x$")
plt.ylabel(r"$v$")
plt.subplot(1,3,2)
mse_lstm = np.mean((gd_smsd-lstm_smsd)**2)
plt.plot(gd_smsd[:,:,0],gd_smsd[:,:,1],label="Ground truth",color="blue",linestyle="-")
plt.plot(lstm_smsd[:300,:,0],lstm_smsd[:300,:,1],label="Model training",color="red",linestyle="--")
# for the prediction, we plot in two steps first 300 steps for on the training range and the last 100 steps for testing
plt.plot(lstm_smsd[300:,:,0],lstm_smsd[300:,:,1],label=r"Model testing($\epsilon$={:.1e})".format(mse_lstm),color="red",linestyle=":")
plt.legend(loc="upper right",frameon=False)

plt.grid()
plt.xlabel(r"$x$")
plt.ylabel(r"$v$")
plt.text(0.5, vertical_position, subplot_labels[1], ha='center', va='top', fontsize=10, transform=plt.gca().transAxes)
plt.subplot(1,3,3)
mse_fcnn = np.mean((gd_smsd-fcnn_smsd)**2)
plt.plot(gd_smsd[:,:,0],gd_smsd[:,:,1],label="Ground truth",color="blue",linestyle="-")
plt.plot(fcnn_smsd[:300,:,0],fcnn_smsd[:300,:,1],label="Model training",color="red",linestyle="--")
# for the prediction, we plot in two steps first 300 steps for on the training range and the last 100 steps for testing
plt.plot(fcnn_smsd[300:,:,0],fcnn_smsd[300:,:,1],label=r"Model testing($\epsilon$={:.1e})".format(mse_fcnn),color="red",linestyle=":")
plt.legend(loc="upper right",frameon=False)
plt.grid()
plt.xlabel(r"$x$")
plt.ylabel(r"$v$")
plt.text(0.5, vertical_position, subplot_labels[2], ha='center', va='top', fontsize=10, transform=plt.gca().transAxes)
plt.tight_layout()
#plt.savefig("figures/Single_Mass_Spring_Damper/smsd_phase_space.png")




#plt.savefig("figures/smsd_phase_space.png"
plt.show()