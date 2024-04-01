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

#load the data to plot the figures
gd_sms=np.load("saved_data/Single_Mass_Spring/Ground_truth for Single_Mass_Spring with training_size=300_num_steps_test=3000_dt=0.01.npy")
pnode_rk4_sms=np.load("saved_data/Single_Mass_Spring/PNODE_RK4 for Single_Mass_Spring with training_size=300_num_steps_test=3000_dt=0.01.npy")
pnode_sms_sym=np.load("saved_data/Single_Mass_Spring_Symplectic/PNODE_Symplectic for Single_Mass_Spring_Symplectic with training_size=300_num_steps_test=3000_dt=0.01.npy")
rk4_sms=np.load("saved_data/Single_Mass_Spring/RK4 for Single_Mass_Spring with training_size=300_num_steps_test=3000_dt=0.01.npy")
lnn_sms=np.load("saved_data/Single_Mass_Spring/LNN for Single_Mass_Spring with training_size=300_num_steps_test=3000_dt=0.01.npy")
hnn_sms=np.load("saved_data/Single_Mass_Spring_Symplectic/HNN for Single_Mass_Spring_Symplectic with training_size=300_num_steps_test=3000_dt=0.01.npy")
pnode_rk4_sms=pnode_rk4_sms[0:3000,:,:]
lnn_sms=lnn_sms.reshape((3000,1,2))
hnn_sms=hnn_sms.reshape((3000,1,2))

pnode_sms_sym[:,:,1]= pnode_sms_sym[:,:,1]/10
hnn_sms[:,:,1]=hnn_sms [:,:,1]/10
print("gd_sms shape:",gd_sms.shape)
print("pnode_rk4_sms shape:",pnode_rk4_sms.shape)
print("pnode_sms_sym shape:",pnode_sms_sym.shape)
print("rk4_sms shape:",rk4_sms.shape)
print("lnn_sms shape:",lnn_sms.shape)
print("hnn_sms2 shape:",hnn_sms.shape)

t_train=np.arange(0,3,0.01)
t_test=np.arange(0,30,0.01)
num_split=300

subplot_labels = ['(a): MNODE_RK4', '(b): MNODE_RK4', '(c): MNODE_LF', '(d): MNODE_LF', '(e): RK4', '(f): RK4', '(g): LNN', '(h): LNN', '(i): HNN', '(j): HNN']

mse_pnode_rk4=np.mean((gd_sms-pnode_rk4_sms)**2)
mse_pnode_lf=np.mean((gd_sms-pnode_sms_sym)**2)
mse_rk4=np.mean((gd_sms-rk4_sms)**2)
mse_lnn=np.mean((gd_sms-lnn_sms)**2)
mse_hnn=np.mean((gd_sms-hnn_sms)**2)
print("mse_pnode_rk4:",mse_pnode_rk4)
print("mse_pnode_lf:",mse_pnode_lf)
print("mse_rk4:",mse_rk4)
print("mse_lnn:",mse_lnn)
print("mse_hnn:",mse_hnn)


label_fontsize = 15
legend_fontsize = 15
# Vertical position of the label below the x-axis
vertical_position = -0.28  # Adjust this as needed
fig,axs=plt.subplots(5,2,figsize=(10,14))
axs[0, 0].plot(t_test, gd_sms[:,:,0], 'b-', label="Ground truth")
axs[0, 0].plot(t_test[:300], pnode_rk4_sms[:300,:,0], 'r--', label="Model training")
axs[0, 0].plot(t_test[300:], pnode_rk4_sms[300:,:,0], 'r:', label="Model testing")
axs[0, 0].grid()
axs[0, 0].set_xlabel(r"$t$",fontsize=label_fontsize)
axs[0, 0].set_ylabel(r"$x$",fontsize=label_fontsize)
axs[0, 0].text(0.5, vertical_position, subplot_labels[0], ha='center', va='top', fontsize=label_fontsize, transform=axs[0, 0].transAxes)

axs[0, 1].plot(t_test, gd_sms[:,:,1], 'b-', label="Ground truth")
axs[0, 1].plot(t_test[:300], pnode_rk4_sms[:300,:,1], 'r--', label="Model training")
axs[0, 1].plot(t_test[300:], pnode_rk4_sms[300:,:,1], 'r:', label="Model testing")
axs[0, 1].grid()
axs[0, 1].set_xlabel(r"$t$",fontsize=label_fontsize)
axs[0, 1].set_ylabel(r"$v$",fontsize=label_fontsize)
axs[0, 1].text(0.5, vertical_position, subplot_labels[1], ha='center', va='top', fontsize=label_fontsize, transform=axs[0, 1].transAxes)
axs[1, 0].plot(t_test, gd_sms[:,:,0], 'b-', label="Ground truth")
axs[1, 0].plot(t_test[:300], pnode_sms_sym[:300,:,0], 'r--', label="Model training")
axs[1, 0].plot(t_test[300:], pnode_sms_sym[300:,:,0], 'r:', label="Model testing")
axs[1, 0].grid()
axs[1, 0].set_xlabel(r"$t$",fontsize=label_fontsize)
axs[1, 0].set_ylabel(r"$x$",fontsize=label_fontsize)
axs[1, 0].text(0.5, vertical_position, subplot_labels[2], ha='center', va='top', fontsize=label_fontsize, transform=axs[1, 0].transAxes)

axs[1, 1].plot(t_test, gd_sms[:,:,1], 'b-', label="Ground truth")
axs[1, 1].plot(t_test[:300], pnode_sms_sym[:300,:,1], 'r--', label="Model training")
axs[1, 1].plot(t_test[300:], pnode_sms_sym[300:,:,1], 'r:', label="Model testing")
axs[1, 1].grid()
axs[1, 1].set_xlabel(r"$t$",fontsize=label_fontsize)
axs[1, 1].set_ylabel(r"$v$",fontsize=label_fontsize)
axs[1, 1].text(0.5, vertical_position, subplot_labels[3], ha='center', va='top', fontsize=label_fontsize, transform=axs[1, 1].transAxes)

axs[2, 0].plot(t_test, gd_sms[:,:,0], 'b-', label="Ground truth")
axs[2, 0].plot(t_test[:300], rk4_sms[:300,:,0], 'r--', label="Model training")
axs[2, 0].plot(t_test[300:], rk4_sms[300:,:,0], 'r:', label="Model testing")
axs[2, 0].grid()
axs[2, 0].set_xlabel(r"$t$",fontsize=label_fontsize)
axs[2, 0].set_ylabel(r"$x$",fontsize=label_fontsize)
axs[2, 0].text(0.5, vertical_position, subplot_labels[4], ha='center', va='top', fontsize=label_fontsize, transform=axs[2, 0].transAxes)

axs[2, 1].plot(t_test, gd_sms[:,:,1], 'b-', label="Ground truth")
axs[2, 1].plot(t_test[:300], rk4_sms[:300,:,1], 'r--', label="Model training")
axs[2, 1].plot(t_test[300:], rk4_sms[300:,:,1], 'r:', label="Model testing")
axs[2, 1].grid()
axs[2, 1].set_xlabel(r"$t$",fontsize=label_fontsize)
axs[2, 1].set_ylabel(r"$v$",fontsize=label_fontsize)
axs[2, 1].text(0.5, vertical_position, subplot_labels[5], ha='center', va='top', fontsize=label_fontsize, transform=axs[2, 1].transAxes)

axs[3, 0].plot(t_test, gd_sms[:,:,0], 'b-', label="Ground truth")
axs[3, 0].plot(t_test[:300], lnn_sms[:300,:,0], 'r--', label="Model training")
axs[3, 0].plot(t_test[300:], lnn_sms[300:,:,0], 'r:', label="Model testing")
axs[3, 0].grid()
axs[3, 0].set_xlabel(r"$t$",fontsize=label_fontsize)
axs[3, 0].set_ylabel(r"$x$",fontsize=label_fontsize)
axs[3, 0].text(0.5, vertical_position, subplot_labels[6], ha='center', va='top', fontsize=label_fontsize, transform=axs[3, 0].transAxes)

axs[3, 1].plot(t_test, gd_sms[:,:,1], 'b-', label="Ground truth")
axs[3, 1].plot(t_test[:300], lnn_sms[:300,:,1], 'r--', label="Model training")
axs[3, 1].plot(t_test[300:], lnn_sms[300:,:,1], 'r:', label="Model testing")
axs[3, 1].grid()
axs[3, 1].set_xlabel(r"$t$",fontsize=label_fontsize)
axs[3, 1].set_ylabel(r"$v$",fontsize=label_fontsize)
axs[3, 1].text(0.5, vertical_position, subplot_labels[7], ha='center', va='top', fontsize=label_fontsize, transform=axs[3, 1].transAxes)

axs[4, 0].plot(t_test, gd_sms[:,:,0], 'b-', label="Ground truth")
axs[4, 0].plot(t_test[:300], hnn_sms[:300,:,0], 'r--', label="Model training")
axs[4, 0].plot(t_test[300:], hnn_sms[300:,:,0], 'r:', label="Model testing")
axs[4, 0].grid()
axs[4, 0].set_xlabel(r"$t$",fontsize=label_fontsize)
axs[4, 0].set_ylabel(r"$x$",fontsize=label_fontsize)
axs[4, 0].text(0.5, vertical_position, subplot_labels[8], ha='center', va='top', fontsize=label_fontsize, transform=axs[4, 0].transAxes)

axs[4, 1].plot(t_test, gd_sms[:,:,1], 'b-', label="Ground truth")
axs[4, 1].plot(t_test[:300], hnn_sms[:300,:,1], 'r--', label="Model training")
axs[4, 1].plot(t_test[300:], hnn_sms[300:,:,1], 'r:', label="Model testing")
axs[4, 1].grid()
axs[4, 1].set_xlabel(r"$t$",fontsize=label_fontsize)
axs[4, 1].set_ylabel(r"$v$",fontsize=label_fontsize)
axs[4, 1].text(0.5, vertical_position, subplot_labels[9], ha='center', va='top', fontsize=label_fontsize, transform=axs[4, 1].transAxes)

handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=3, fontsize=legend_fontsize)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("figures/Single_Mass_Spring/sms_x_v_time.png")




subplot_labels = ['(a)', '(b)']
vertical_position = -0.13


fig3,axs3=plt.subplots(1,2,figsize=(10,6))
axs3[0].plot(gd_sms[:,0,0],gd_sms[:,0,1],label="Ground truth",linestyle='-',color='black')
axs3[0].plot(pnode_rk4_sms[:,0,0],pnode_rk4_sms[:,0,1],label="MNODE_RK4",linestyle='--',color='red')
axs3[0].plot(rk4_sms[:,0,0],rk4_sms[:,0,1],label="RK4",linestyle='-.',color='blue')
axs3[0].plot(pnode_sms_sym[:,0,0],pnode_sms_sym[:,0,1],label="MNODE_LF",linestyle=':',color='green')
axs3[0].plot(lnn_sms[:,0,0],lnn_sms[:,0,1],label="LNN",linestyle='-.',color='orange')
axs3[0].plot(hnn_sms[:,0,0],hnn_sms[:,0,1],label="HNN",linestyle=':',color='purple')
axs3[0].grid()
axs3[0].text(0.5, vertical_position, subplot_labels[0], ha='center', va='top', fontsize=label_fontsize, transform=axs3[0].transAxes)
#plt.title("Single Mass Spring")
axs3[0].set_xlabel("$x$",fontsize=label_fontsize)
axs3[0].set_ylabel("$v$",fontsize=label_fontsize)
axs3[1].plot(t_test,energy_sms(gd_sms),label="Ground truth",linestyle='-',color='black')
axs3[1].plot(t_test,energy_sms(pnode_rk4_sms),label="MNODE_RK4",linestyle='--',color='red')
axs3[1].plot(t_test,energy_sms(rk4_sms),label="RK4",linestyle='-.',color='blue')
axs3[1].plot(t_test,energy_sms(pnode_sms_sym),label="MNODE_LF",linestyle=':',color='green')
axs3[1].plot(t_test,energy_sms(lnn_sms),label="LNN",linestyle='-.',color='orange')
axs3[1].plot(t_test,energy_sms(hnn_sms),label="HNN",linestyle=':',color='purple')
axs3[1].grid()
axs3[1].text(0.5, vertical_position, subplot_labels[1], ha='center', va='top', fontsize=label_fontsize, transform=axs3[1].transAxes)
axs3[1].set_xlabel("$t$",fontsize=label_fontsize)
axs3[1].set_ylabel("$E$",fontsize=label_fontsize)
handles, labels = axs3[0].get_legend_handles_labels()
fig3.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.01), ncol=3, fontsize=legend_fontsize)
fig3.tight_layout(rect=[0, 0, 1, 0.9])

plt.savefig("figures/Single_Mass_Spring/sms_phase_space_energy.png")



plt.show()
