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


gd_dp=np.load("saved_data/Double_Pendulum/Ground_truth for Double_Pendulum with training_size=300_num_steps_test=400_dt=0.01.npy")
pnode_dp=np.load("saved_data/Double_Pendulum/PNODE for Double_Pendulum with training_size=300_num_steps_test=400_dt=0.01.npy")
lstm_dp=np.load("saved_data/Double_Pendulum/LSTM for Double_Pendulum with training_size=300_num_steps_test=400_dt=0.01.npy")
fcnn_dp=np.load("saved_data/Double_Pendulum/FCNN for Double_Pendulum with training_size=300_num_steps_test=400_dt=0.01.npy")

pnode_dp=pnode_dp[0:400]
lstm_dp=lstm_dp[0:400]
print("gd_dp shape:",gd_dp.shape)
print("pnode_dp shape:",pnode_dp.shape)
print("lstm_dp shape:",lstm_dp.shape)
print("fcnn_dp shape:",fcnn_dp.shape)

t_test = np.linspace(0, 0.01*400, 400)

#plot the trajectory for the double pendulum
plt.figure(figsize=(10,10))
plt.plot(gd_dp[:,:,0],gd_dp[:,:,1],label="Ground truth")
plt.plot(pnode_dp[:,:,0],pnode_dp[:,:,1],label="MNODE",linestyle="--")
#plt.plot(lstm_dp[:,:,0],lstm_dp[:,:,1],label="LSTM")
#plt.plot(fcnn_dp[:,:,0],fcnn_dp[:,:,1],label="FCNN")
plt.legend(loc="upper left", frameon=False)
plt.title("Double Pendulum")
plt.xlabel("$t$")
plt.ylabel(r"$\theta$")
#plt.savefig("saved_figures/Double_Pendulum.png")
plt.show()

subplot_labels = ['(a): MNODE', '(b): MNODE', '(c): LSTM', '(d): LSTM', '(e): FCNN', '(f): FCNN']

vertical_position = -0.23  # Adjust this as needed
# the legend should contrain the mean square error
mse_pnode = np.mean((gd_dp-pnode_dp)**2)
mse_lstm = np.mean((gd_dp-lstm_dp)**2)
mse_fcnn = np.mean((gd_dp-fcnn_dp)**2)
#print the mse for each model
print("mse_pnode:",mse_pnode)
print("mse_lstm:",mse_lstm)
print("mse_fcnn:",mse_fcnn)
label_fontsize = 15
legend_fontsize = 15
##using unifed legend outside the plot for the six subplots
fig,axs=plt.subplots(3,2,figsize=(10,10))
axs[0,0].plot(t_test,gd_dp[:,0,0],label="Ground truth",color="blue",linestyle="-")
axs[0,0].plot(t_test,gd_dp[:,1,0],color="blue",linestyle="-")
axs[0,0].plot(t_test[0:300],pnode_dp[:300,0,0],label="Model training",color="red",linestyle="--")
axs[0,0].plot(t_test[0:300],pnode_dp[:300,1,0],color="red",linestyle="--")
axs[0,0].plot(t_test[300:],pnode_dp[300:,0,0],label="Model testing",color="red",linestyle=":")
axs[0,0].plot(t_test[300:],pnode_dp[300:,1,0],color="red",linestyle=":")
axs[0,0].grid()
axs[0,0].set_xlabel(r"$t$",fontsize=label_fontsize)
axs[0,0].set_ylabel(r"$\theta$",fontsize=label_fontsize)
axs[0,0].text(0.5, vertical_position, subplot_labels[0], ha='center', va='top', fontsize=label_fontsize, transform=axs[0, 0].transAxes)
axs[0,1].plot(t_test,gd_dp[:,0,1],label="Ground truth training",color="blue",linestyle="-")
axs[0,1].plot(t_test,gd_dp[:,1,1],color="blue",linestyle="-")
axs[0,1].plot(t_test[0:300],pnode_dp[:300,0,1],label="Model training",color="red",linestyle="--")
axs[0,1].plot(t_test[0:300],pnode_dp[:300,1,1],color="red",linestyle="--")
axs[0,1].plot(t_test[300:],pnode_dp[300:,0,1],label="Model testing",color="red",linestyle=":")
axs[0,1].plot(t_test[300:],pnode_dp[300:,1,1],color="red",linestyle=":")
axs[0,1].grid()
axs[0,1].set_xlabel(r"$t$",fontsize=label_fontsize)
axs[0,1].set_ylabel(r"$\omega$",fontsize=label_fontsize)
axs[0,1].text(0.5, vertical_position, subplot_labels[1], ha='center', va='top', fontsize=label_fontsize, transform=axs[0, 1].transAxes)
axs[1,0].plot(t_test,gd_dp[:,0,0],label="Ground truth",color="blue",linestyle="-")
axs[1,0].plot(t_test,gd_dp[:,1,0],color="blue",linestyle="-")
axs[1,0].plot(t_test[0:300],lstm_dp[:300,0,0],label="Model training",color="red",linestyle="--")
axs[1,0].plot(t_test[0:300],lstm_dp[:300,1,0],color="red",linestyle="--")
axs[1,0].plot(t_test[300:],lstm_dp[300:,0,0],label="Model testing",color="red",linestyle=":")
axs[1,0].plot(t_test[300:],lstm_dp[300:,1,0],color="red",linestyle=":")
axs[1,0].grid()
axs[1,0].set_xlabel(r"$t$",fontsize=label_fontsize)
axs[1,0].set_ylabel(r"$\theta$",fontsize=label_fontsize)
axs[1,0].text(0.5, vertical_position, subplot_labels[2], ha='center', va='top', fontsize=label_fontsize, transform=axs[1, 0].transAxes)
axs[1,1].plot(t_test,gd_dp[:,0,1],label="Ground truth training",color="blue",linestyle="-")
axs[1,1].plot(t_test,gd_dp[:,1,1],color="blue",linestyle="-")
axs[1,1].plot(t_test[0:300],lstm_dp[:300,0,1],label="Model training",color="red",linestyle="--")
axs[1,1].plot(t_test[0:300],lstm_dp[:300,1,1],color="red",linestyle="--")
axs[1,1].plot(t_test[300:],lstm_dp[300:,0,1],label="Model testing",color="red",linestyle=":")
axs[1,1].plot(t_test[300:],lstm_dp[300:,1,1],color="red",linestyle=":")
axs[1,1].grid()
axs[1,1].set_xlabel(r"$t$",fontsize=label_fontsize)
axs[1,1].set_ylabel(r"$\omega$",fontsize=label_fontsize)
axs[1,1].text(0.5, vertical_position, subplot_labels[3], ha='center', va='top', fontsize=label_fontsize, transform=axs[1, 1].transAxes)
axs[2,0].plot(t_test,gd_dp[:,0,0],label="Ground truth",color="blue",linestyle="-")
axs[2,0].plot(t_test,gd_dp[:,1,0],color="blue",linestyle="-")
axs[2,0].plot(t_test[0:300],fcnn_dp[:300,0,0],label="Model training",color="red",linestyle="--")
axs[2,0].plot(t_test[0:300],fcnn_dp[:300,1,0],color="red",linestyle="--")
axs[2,0].plot(t_test[300:],fcnn_dp[300:,0,0],label="Model testing",color="red",linestyle=":")
axs[2,0].plot(t_test[300:],fcnn_dp[300:,1,0],color="red",linestyle=":")
axs[2,0].grid()
axs[2,0].set_xlabel(r"$t$",fontsize=label_fontsize)
axs[2,0].set_ylabel(r"$\theta$",fontsize=label_fontsize)
axs[2,0].text(0.5, vertical_position, subplot_labels[4], ha='center', va='top', fontsize=label_fontsize, transform=axs[2, 0].transAxes)
axs[2,1].plot(t_test,gd_dp[:,0,1],label="Ground truth training",color="blue",linestyle="-")
axs[2,1].plot(t_test,gd_dp[:,1,1],color="blue",linestyle="-")
axs[2,1].plot(t_test[0:300],fcnn_dp[:300,0,1],label="Model training",color="red",linestyle="--")
axs[2,1].plot(t_test[0:300],fcnn_dp[:300,1,1],color="red",linestyle="--")
axs[2,1].plot(t_test[300:],fcnn_dp[300:,0,1],label="Model testing",color="red",linestyle=":")
axs[2,1].plot(t_test[300:],fcnn_dp[300:,1,1],color="red",linestyle=":")
axs[2,1].grid()
axs[2,1].set_xlabel(r"$t$",fontsize=label_fontsize)
axs[2,1].set_ylabel(r"$\omega$",fontsize=label_fontsize)
axs[2,1].text(0.5, vertical_position, subplot_labels[5], ha='center', va='top', fontsize=label_fontsize, transform=axs[2, 1].transAxes)
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.99), ncol=3, fontsize=legend_fontsize)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("figures/Double_Pendulum/dp_x_v_time.png")





#subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

#also change the legend for the phase space plot to be on the top
fig2,axs2=plt.subplots(3,2,figsize=(10,10))
axs2[0,0].plot(gd_dp[:,0,0],gd_dp[:,0,1],label="Ground truth",color="blue",linestyle="-")
axs2[0,0].plot(pnode_dp[:300,0,0],pnode_dp[:300,0,1],label="Model training",color="red",linestyle="--")
axs2[0,0].plot(pnode_dp[300:,0,0],pnode_dp[300:,0,1],label="Model testing",color="red",linestyle=":")
axs2[0,0].grid()
axs2[0,0].set_xlabel(r"$\theta$",fontsize=label_fontsize)
axs2[0,0].set_ylabel(r"$\omega$",fontsize=label_fontsize)
axs2[0,0].text(0.5, vertical_position, subplot_labels[0], ha='center', va='top', fontsize=label_fontsize, transform=axs2[0, 0].transAxes)
axs2[0,1].plot(gd_dp[:,1,0],gd_dp[:,1,1],label="Ground truth",color="blue",linestyle="-")
axs2[0,1].plot(pnode_dp[:300,1,0],pnode_dp[:300,1,1],label="Model training",color="red",linestyle="--")
axs2[0,1].plot(pnode_dp[300:,1,0],pnode_dp[300:,1,1],label="Model testing",color="red",linestyle=":")
axs2[0,1].grid()
axs2[0,1].set_xlabel(r"$\theta$",fontsize=label_fontsize)
axs2[0,1].set_ylabel(r"$\omega$",fontsize=label_fontsize)
axs2[0,1].text(0.5, vertical_position, subplot_labels[1], ha='center', va='top', fontsize=label_fontsize, transform=axs2[0, 1].transAxes)
axs2[1,0].plot(gd_dp[:,0,0],gd_dp[:,0,1],label="Ground truth",color="blue",linestyle="-")
axs2[1,0].plot(lstm_dp[:300,0,0],lstm_dp[:300,0,1],label="Model training",color="red",linestyle="--")
axs2[1,0].plot(lstm_dp[300:,0,0],lstm_dp[300:,0,1],label="Model testing",color="red",linestyle=":")
axs2[1,0].grid()
axs2[1,0].set_xlabel(r"$\theta$",fontsize=label_fontsize)
axs2[1,0].set_ylabel(r"$\omega$",fontsize=label_fontsize)
axs2[1,0].text(0.5, vertical_position, subplot_labels[2], ha='center', va='top', fontsize=label_fontsize, transform=axs2[1, 0].transAxes)
axs2[1,1].plot(gd_dp[:,1,0],gd_dp[:,1,1],label="Ground truth",color="blue",linestyle="-")
axs2[1,1].plot(lstm_dp[:300,1,0],lstm_dp[:300,1,1],label="Model training",color="red",linestyle="--")
axs2[1,1].plot(lstm_dp[300:,1,0],lstm_dp[300:,1,1],label="Model testing",color="red",linestyle=":")
axs2[1,1].grid()
axs2[1,1].set_xlabel(r"$\theta$",fontsize=label_fontsize)
axs2[1,1].set_ylabel(r"$\omega$",fontsize=label_fontsize)
axs2[1,1].text(0.5, vertical_position, subplot_labels[3], ha='center', va='top', fontsize=label_fontsize, transform=axs2[1, 1].transAxes)
axs2[2,0].plot(gd_dp[:,0,0],gd_dp[:,0,1],label="Ground truth",color="blue",linestyle="-")
axs2[2,0].plot(fcnn_dp[:300,0,0],fcnn_dp[:300,0,1],label="Model training",color="red",linestyle="--")
axs2[2,0].plot(fcnn_dp[300:,0,0],fcnn_dp[300:,0,1],label="Model testing",color="red",linestyle=":")
axs2[2,0].grid()
axs2[2,0].set_xlabel(r"$\theta$",fontsize=label_fontsize)
axs2[2,0].set_ylabel(r"$\omega$",fontsize=label_fontsize)
axs2[2,0].text(0.5, vertical_position, subplot_labels[4], ha='center', va='top', fontsize=label_fontsize, transform=axs2[2, 0].transAxes)
axs2[2,1].plot(gd_dp[:,1,0],gd_dp[:,1,1],label="Ground truth",color="blue",linestyle="-")
axs2[2,1].plot(fcnn_dp[:300,1,0],fcnn_dp[:300,1,1],label="Model training",color="red",linestyle="--")
axs2[2,1].plot(fcnn_dp[300:,1,0],fcnn_dp[300:,1,1],label="Model testing",color="red",linestyle=":")
axs2[2,1].grid()
axs2[2,1].set_xlabel(r"$\theta$",fontsize=label_fontsize)
axs2[2,1].set_ylabel(r"$\omega$",fontsize=label_fontsize)
axs2[2,1].text(0.5, vertical_position, subplot_labels[5], ha='center', va='top', fontsize=label_fontsize, transform=axs2[2, 1].transAxes)
handles, labels = axs2[0, 0].get_legend_handles_labels()
fig2.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.99), ncol=3, fontsize=legend_fontsize)
fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("figures/Double_Pendulum/dp_x_v.png")

plt.show()

