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


gd_tms=np.load("saved_data/Triple_Mass_Spring_Damper/Ground_truth for Triple_Mass_Spring_Damper with training_size=300_num_steps_test=400_dt=0.01.npy ")
pnode_tms=np.load("saved_data/Triple_Mass_Spring_Damper/PNODE for Triple_Mass_Spring_Damper with training_size=300_num_steps_test=400_dt=0.01.npy")
lstm_tms=np.load("saved_data/Triple_Mass_Spring_Damper/LSTM for Triple_Mass_Spring_Damper with training_size=300_num_steps_test=400_dt=0.01.npy")
fcnn_tms=np.load("saved_data/Triple_Mass_Spring_Damper/FCNN for Triple_Mass_Spring_Damper with training_size=300_num_steps_test=400_dt=0.01.npy")

pnode_tms=pnode_tms[0:400,:,:]
lstm_tms=lstm_tms[0:400,:,:]
print("gd_tms shape:",gd_tms.shape)
print("pnode_tms shape:",pnode_tms.shape)
print("lstm_tms shape:",lstm_tms.shape)
print("fcnn_tms shape:",fcnn_tms.shape)

t_test = np.linspace(0, 0.01*400, 400)
t_train= np.linspace(0, 0.01*300, 300)


plt.figure(figsize=(10,10))
plt.plot(gd_tms[:,:,0],gd_tms[:,:,1],label="Ground truth")
plt.plot(pnode_tms[:,:,0],pnode_tms[:,:,1],label="MNODE",linestyle='--')
#plt.plot(lstm_tms[:,:,0],lstm_tms[:,:,1],label="LSTM")
#plt.plot(fcnn_tms[:,:,0],fcnn_tms[:,:,1],label="FCNN")
plt.legend()
plt.title("Triple Mass Spring Damper")
plt.xlabel("t")
plt.ylabel("x")
#plt.savefig("saved_figures/Triple_Mass_Spring_Damper.png")
plt.show()

#like the plot for the single mass spring damper, we plot the x-t, v-t for the triple mass spring damper
# the only difference is we have three bodies for the triple mass spring damper
# we plot the three bodies in the same plot, the legend will only appear once

# make a plot with multiple subplots
# scientific notation!!
# scientific style!!
# Define the labels for each subplot
subplot_labels = ['(a): MNODE', '(b): MNODE', '(c): LSTM', '(d): LSTM', '(e): FCNN', '(f): FCNN']

# Vertical position of the label below the x-axis
label_fontsize = 15
legend_fontsize = 15
# Vertical position of the label below the x-axis
vertical_position = -0.22  # Adjust this as needed
mse_pnode = np.mean((gd_tms-pnode_tms)**2)
mse_lstm = np.mean((gd_tms-lstm_tms)**2)
mse_fcnn = np.mean((gd_tms-fcnn_tms)**2)
#print the mse for the three models
print("mse_pnode:",mse_pnode)
print("mse_lstm:",mse_lstm)
print("mse_fcnn:",mse_fcnn)

#using unifed legend outside the plot for the six subplots
fig,axs=plt.subplots(3,2,figsize=(10,10))
axs[0,0].plot(t_test,gd_tms[:,0,0],label="Ground truth",color="blue",linestyle="-")
axs[0,0].plot(t_test,gd_tms[:,1,0],color="blue",linestyle="-")
axs[0,0].plot(t_test,gd_tms[:,2,0],color="blue",linestyle="-")

axs[0,0].plot(t_test[:300:],pnode_tms[:300,0,0],label="Model training",color="red",linestyle="--")
axs[0,0].plot(t_test[:300:],pnode_tms[:300,1,0],color="red",linestyle="--")
axs[0,0].plot(t_test[:300:],pnode_tms[:300,2,0],color="red",linestyle="--")
axs[0,0].plot(t_test[300:],pnode_tms[300:,0,0],label=r"Model testing",color="red",linestyle=":")
axs[0,0].plot(t_test[300:],pnode_tms[300:,1,0],color="red",linestyle=":")
axs[0,0].plot(t_test[300:],pnode_tms[300:,2,0],color="red",linestyle=":")
axs[0,0].grid()
axs[0,0].set_xlabel("$t$",fontsize=label_fontsize)
axs[0,0].set_ylabel("$x$",fontsize=label_fontsize)
axs[0,0].text(0.5, vertical_position, subplot_labels[0], ha='center', va='top', fontsize=label_fontsize, transform=axs[0,0].transAxes)

axs[0,1].plot(t_test,gd_tms[:,0,1],label="Ground truth",color="blue",linestyle="-")
axs[0,1].plot(t_test,gd_tms[:,1,1],color="blue",linestyle="-")
axs[0,1].plot(t_test,gd_tms[:,2,1],color="blue",linestyle="-")

axs[0,1].plot(t_test[:300:],pnode_tms[:300,0,1],label="Model training",color="red",linestyle="--")
axs[0,1].plot(t_test[:300:],pnode_tms[:300,1,1],color="red",linestyle="--")
axs[0,1].plot(t_test[:300:],pnode_tms[:300,2,1],color="red",linestyle="--")
axs[0,1].plot(t_test[300:],pnode_tms[300:,0,1],label=r"Model testing",color="red",linestyle=":")
axs[0,1].plot(t_test[300:],pnode_tms[300:,1,1],color="red",linestyle=":")
axs[0,1].plot(t_test[300:],pnode_tms[300:,2,1],color="red",linestyle=":")
axs[0,1].grid()
axs[0,1].set_xlabel("$t$",fontsize=label_fontsize)
axs[0,1].set_ylabel("$v$",fontsize=label_fontsize)
axs[0,1].text(0.5, vertical_position, subplot_labels[1], ha='center', va='top', fontsize=label_fontsize, transform=axs[0,1].transAxes)

axs[1,0].plot(t_test,gd_tms[:,0,0],label="Ground truth",color="blue",linestyle="-")
axs[1,0].plot(t_test,gd_tms[:,1,0],color="blue",linestyle="-")
axs[1,0].plot(t_test,gd_tms[:,2,0],color="blue",linestyle="-")

axs[1,0].plot(t_test[:300:],lstm_tms[:300,0,0],label="Model training",color="red",linestyle="--")
axs[1,0].plot(t_test[:300:],lstm_tms[:300,1,0],color="red",linestyle="--")
axs[1,0].plot(t_test[:300:],lstm_tms[:300,2,0],color="red",linestyle="--")
axs[1,0].plot(t_test[300:],lstm_tms[300:,0,0],label=r"Model testing",color="red",linestyle=":")
axs[1,0].plot(t_test[300:],lstm_tms[300:,1,0],color="red",linestyle=":")
axs[1,0].plot(t_test[300:],lstm_tms[300:,2,0],color="red",linestyle=":")
axs[1,0].grid()
axs[1,0].set_xlabel("$t$",fontsize=label_fontsize)
axs[1,0].set_ylabel("$x$",fontsize=label_fontsize)
axs[1,0].text(0.5, vertical_position, subplot_labels[2], ha='center', va='top', fontsize=label_fontsize, transform=axs[1,0].transAxes)

axs[1,1].plot(t_test,gd_tms[:,0,1],label="Ground truth",color="blue",linestyle="-")
axs[1,1].plot(t_test,gd_tms[:,1,1],color="blue",linestyle="-")
axs[1,1].plot(t_test,gd_tms[:,2,1],color="blue",linestyle="-")

axs[1,1].plot(t_test[:300:],lstm_tms[:300,0,1],label="Model training",color="red",linestyle="--")
axs[1,1].plot(t_test[:300:],lstm_tms[:300,1,1],color="red",linestyle="--")
axs[1,1].plot(t_test[:300:],lstm_tms[:300,2,1],color="red",linestyle="--")
axs[1,1].plot(t_test[300:],lstm_tms[300:,0,1],label=r"Model testing",color="red",linestyle=":")
axs[1,1].plot(t_test[300:],lstm_tms[300:,1,1],color="red",linestyle=":")
axs[1,1].plot(t_test[300:],lstm_tms[300:,2,1],color="red",linestyle=":")
axs[1,1].grid()
axs[1,1].set_xlabel("$t$",fontsize=label_fontsize)
axs[1,1].set_ylabel("$v$",fontsize=label_fontsize)

axs[1,1].text(0.5, vertical_position, subplot_labels[3], ha='center', va='top', fontsize=label_fontsize, transform=axs[1,1].transAxes)

axs[2,0].plot(t_test,gd_tms[:,0,0],label="Ground truth",color="blue",linestyle="-")
axs[2,0].plot(t_test,gd_tms[:,1,0],color="blue",linestyle="-")
axs[2,0].plot(t_test,gd_tms[:,2,0],color="blue",linestyle="-")

axs[2,0].plot(t_test[:300:],fcnn_tms[:300,0,0],label="Model training",color="red",linestyle="--")
axs[2,0].plot(t_test[:300:],fcnn_tms[:300,1,0],color="red",linestyle="--")
axs[2,0].plot(t_test[:300:],fcnn_tms[:300,2,0],color="red",linestyle="--")
axs[2,0].plot(t_test[300:],fcnn_tms[300:,0,0],label=r"Model testing",color="red",linestyle=":")
axs[2,0].plot(t_test[300:],fcnn_tms[300:,1,0],color="red",linestyle=":")
axs[2,0].plot(t_test[300:],fcnn_tms[300:,2,0],color="red",linestyle=":")
axs[2,0].grid()
axs[2,0].set_xlabel("$t$",fontsize=label_fontsize)
axs[2,0].set_ylabel("$x$",fontsize=label_fontsize)

axs[2,0].text(0.5, vertical_position, subplot_labels[4], ha='center', va='top', fontsize=label_fontsize, transform=axs[2,0].transAxes)

axs[2,1].plot(t_test,gd_tms[:,0,1],label="Ground truth",color="blue",linestyle="-")
axs[2,1].plot(t_test,gd_tms[:,1,1],color="blue",linestyle="-")
axs[2,1].plot(t_test,gd_tms[:,2,1],color="blue",linestyle="-")

axs[2,1].plot(t_test[:300:],fcnn_tms[:300,0,1],label="Model training",color="red",linestyle="--")
axs[2,1].plot(t_test[:300:],fcnn_tms[:300,1,1],color="red",linestyle="--")
axs[2,1].plot(t_test[:300:],fcnn_tms[:300,2,1],color="red",linestyle="--")
axs[2,1].plot(t_test[300:],fcnn_tms[300:,0,1],label=r"Model testing",color="red",linestyle=":")
axs[2,1].plot(t_test[300:],fcnn_tms[300:,1,1],color="red",linestyle=":")
axs[2,1].plot(t_test[300:],fcnn_tms[300:,2,1],color="red",linestyle=":")
axs[2,1].grid()
axs[2,1].set_xlabel("$t$",fontsize=label_fontsize)
axs[2,1].set_ylabel("$v$",fontsize=label_fontsize)
axs[2,1].text(0.5, vertical_position, subplot_labels[5], ha='center', va='top', fontsize=label_fontsize, transform=axs[2,1].transAxes)


handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.99), ncol=3, fontsize=legend_fontsize)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("figures/Triple_Mass_Spring_Damper/tmsd_x_v_t.png")



subplot_labels = ['(a): MNODE', '(b): MNODE', '(c): MNODE', '(d): LSTM', '(e): LSTM', '(f): LSTM', '(g): FCNN', '(h): FCNN', '(i): FCNN']

# Vertical position of the label below the x-axis
vertical_position = -0.23  # Adjust this as needed

#also move the legend outside the plot for the phase space plot
fig2,axs2=plt.subplots(3,3,figsize=(10,10))
axs2[0,0].plot(gd_tms[:,0,0],gd_tms[:,0,1],color="blue",label="Ground truth")
axs2[0,0].plot(pnode_tms[:300,0,0],pnode_tms[:300,0,1],label="Model training",color="red",linestyle='--')
axs2[0,0].plot(pnode_tms[300:,0,0],pnode_tms[300:,0,1],label="Model testing",color="red",linestyle=':')
axs2[0,0].grid()
axs2[0,0].set_xlabel("$x$",fontsize=label_fontsize)
axs2[0,0].set_ylabel("$v$",fontsize=label_fontsize)
axs2[0,0].text(0.5, vertical_position, subplot_labels[0], ha='center', va='top', fontsize=label_fontsize, transform=axs2[0,0].transAxes)

axs2[0,1].plot(gd_tms[:,1,0],gd_tms[:,1,1],color="blue",label="Ground truth")
axs2[0,1].plot(pnode_tms[:300,1,0],pnode_tms[:300,1,1],label="Model training",color="red",linestyle='--')
axs2[0,1].plot(pnode_tms[300:,1,0],pnode_tms[300:,1,1],label="Model testing",color="red",linestyle=':')
axs2[0,1].grid()
axs2[0,1].set_xlabel("$x$",fontsize=label_fontsize)
axs2[0,1].set_ylabel("$v$",fontsize=label_fontsize)
axs2[0,1].text(0.5, vertical_position, subplot_labels[1], ha='center', va='top', fontsize=label_fontsize, transform=axs2[0,1].transAxes)

axs2[0,2].plot(gd_tms[:,2,0],gd_tms[:,2,1],color="blue",label="Ground truth")
axs2[0,2].plot(pnode_tms[:300,2,0],pnode_tms[:300,2,1],label="Model training",color="red",linestyle='--')
axs2[0,2].plot(pnode_tms[300:,2,0],pnode_tms[300:,2,1],label="Model testing",color="red",linestyle=':')
axs2[0,2].grid()
axs2[0,2].set_xlabel("$x$",fontsize=label_fontsize)
axs2[0,2].set_ylabel("$v$",fontsize=label_fontsize)
axs2[0,2].text(0.5, vertical_position, subplot_labels[2], ha='center', va='top', fontsize=label_fontsize, transform=axs2[0,2].transAxes)

axs2[1,0].plot(gd_tms[:,0,0],gd_tms[:,0,1],color="blue",label="Ground truth")
axs2[1,0].plot(lstm_tms[:300,0,0],lstm_tms[:300,0,1],label="Model training",color="red",linestyle='--')
axs2[1,0].plot(lstm_tms[300:,0,0],lstm_tms[300:,0,1],label="Model testing",color="red",linestyle=':')
axs2[1,0].grid()
axs2[1,0].set_xlabel("$x$",fontsize=label_fontsize)
axs2[1,0].set_ylabel("$v$",fontsize=label_fontsize)
axs2[1,0].text(0.5, vertical_position, subplot_labels[3], ha='center', va='top', fontsize=label_fontsize, transform=axs2[1,0].transAxes)

axs2[1,1].plot(gd_tms[:,1,0],gd_tms[:,1,1],color="blue",label="Ground truth")
axs2[1,1].plot(lstm_tms[:300,1,0],lstm_tms[:300,1,1],label="Model training",color="red",linestyle='--')
axs2[1,1].plot(lstm_tms[300:,1,0],lstm_tms[300:,1,1],label="Model testing",color="red",linestyle=':')
axs2[1,1].grid()
axs2[1,1].set_xlabel("$x$",fontsize=label_fontsize)
axs2[1,1].set_ylabel("$v$",fontsize=label_fontsize)
axs2[1,1].text(0.5, vertical_position, subplot_labels[4], ha='center', va='top', fontsize=label_fontsize, transform=axs2[1,1].transAxes)

axs2[1,2].plot(gd_tms[:,2,0],gd_tms[:,2,1],color="blue",label="Ground truth")
axs2[1,2].plot(lstm_tms[:300,2,0],lstm_tms[:300,2,1],label="Model training",color="red",linestyle='--')
axs2[1,2].plot(lstm_tms[300:,2,0],lstm_tms[300:,2,1],label="Model testing",color="red",linestyle=':')
axs2[1,2].grid()
axs2[1,2].set_xlabel("$x$",fontsize=label_fontsize)
axs2[1,2].set_ylabel("$v$",fontsize=label_fontsize)
axs2[1,2].text(0.5, vertical_position, subplot_labels[5], ha='center', va='top', fontsize=label_fontsize, transform=axs2[1,2].transAxes)

axs2[2,0].plot(gd_tms[:,0,0],gd_tms[:,0,1],color="blue",label="Ground truth")
axs2[2,0].plot(fcnn_tms[:300,0,0],fcnn_tms[:300,0,1],label="Model training",color="red",linestyle='--')
axs2[2,0].plot(fcnn_tms[300:,0,0],fcnn_tms[300:,0,1],label="Model testing",color="red",linestyle=':')
axs2[2,0].grid()
axs2[2,0].set_xlabel("$x$",fontsize=label_fontsize)
axs2[2,0].set_ylabel("$v$",fontsize=label_fontsize)
axs2[2,0].text(0.5, vertical_position, subplot_labels[6], ha='center', va='top', fontsize=label_fontsize, transform=axs2[2,0].transAxes)

axs2[2,1].plot(gd_tms[:,1,0],gd_tms[:,1,1],color="blue",label="Ground truth")
axs2[2,1].plot(fcnn_tms[:300,1,0],fcnn_tms[:300,1,1],label="Model training",color="red",linestyle='--')
axs2[2,1].plot(fcnn_tms[300:,1,0],fcnn_tms[300:,1,1],label="Model testing",color="red",linestyle=':')
axs2[2,1].grid()
axs2[2,1].set_xlabel("$x$",fontsize=label_fontsize)
axs2[2,1].set_ylabel("$v$",fontsize=label_fontsize)
axs2[2,1].text(0.5, vertical_position, subplot_labels[7], ha='center', va='top', fontsize=label_fontsize, transform=axs2[2,1].transAxes)

axs2[2,2].plot(gd_tms[:,2,0],gd_tms[:,2,1],color="blue",label="Ground truth")
axs2[2,2].plot(fcnn_tms[:300,2,0],fcnn_tms[:300,2,1],label="Model training",color="red",linestyle='--')
axs2[2,2].plot(fcnn_tms[300:,2,0],fcnn_tms[300:,2,1],label="Model testing",color="red",linestyle=':')
axs2[2,2].grid()
axs2[2,2].set_xlabel("$x$",fontsize=label_fontsize)
axs2[2,2].set_ylabel("$v$",fontsize=label_fontsize)
axs2[2,2].text(0.5, vertical_position, subplot_labels[8], ha='center', va='top', fontsize=label_fontsize, transform=axs2[2,2].transAxes)

handles, labels = axs2[0, 0].get_legend_handles_labels()
fig2.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.99), ncol=3, fontsize=legend_fontsize)
fig2.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.savefig("figures/Triple_Mass_Spring_Damper/tmsd_x_v.png")


plt.show()